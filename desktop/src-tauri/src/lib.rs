use serde::Serialize;
#[cfg(not(target_os = "android"))]
use std::io::{BufRead, BufReader, Read, Write};
#[cfg(not(target_os = "android"))]
use std::net::{SocketAddr, TcpListener, TcpStream};
#[cfg(not(target_os = "android"))]
use std::path::{Path, PathBuf};
#[cfg(not(target_os = "android"))]
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
#[cfg(not(target_os = "android"))]
use std::time::{Duration, Instant};
#[cfg(not(target_os = "android"))]
use tauri::{Emitter, Manager};
use tauri::RunEvent;

#[derive(Clone, Serialize)]
pub struct ApiReadyPayload {
    pub port: u16,
    pub base_url: String,
    pub startup_ms: u64,
}

#[derive(Clone, Serialize)]
pub struct SidecarStatusDto {
    pub available: bool,
    pub port: u16,
    pub base_url: String,
    pub ready: bool,
    pub startup_ms: Option<u64>,
    pub error: Option<String>,
}

struct SidecarRuntime {
    available: bool,
    ready: bool,
    startup_ms: Option<u64>,
    error: Option<String>,
    #[cfg(not(target_os = "android"))]
    child: Option<Child>,
    #[cfg(windows)]
    job: Option<win_job::JobObject>,
    #[cfg(not(target_os = "android"))]
    last_stderr: Vec<String>,
}

pub struct SidecarController {
    port: u16,
    base_url: String,
    stop_flag: Arc<AtomicBool>,
    runtime: Mutex<SidecarRuntime>,
}

impl SidecarController {
    fn new(port: u16, available: bool) -> Self {
        let base_url = if available && port > 0 {
            format!("http://127.0.0.1:{}", port)
        } else {
            String::new()
        };
        Self {
            port,
            base_url,
            stop_flag: Arc::new(AtomicBool::new(false)),
            runtime: Mutex::new(SidecarRuntime {
                available,
                ready: false,
                startup_ms: None,
                error: None,
                #[cfg(not(target_os = "android"))]
                child: None,
                #[cfg(windows)]
                job: None,
                #[cfg(not(target_os = "android"))]
                last_stderr: Vec::new(),
            }),
        }
    }

    fn status(&self) -> SidecarStatusDto {
        let rt = self.runtime.lock().expect("sidecar runtime lock poisoned");
        SidecarStatusDto {
            available: rt.available,
            port: self.port,
            base_url: self.base_url.clone(),
            ready: rt.ready,
            startup_ms: rt.startup_ms,
            error: rt.error.clone(),
        }
    }

    #[cfg(not(target_os = "android"))]
    fn push_stderr_line(&self, line: String) {
        if let Ok(mut rt) = self.runtime.lock() {
            if rt.last_stderr.len() >= 25 {
                rt.last_stderr.remove(0);
            }
            rt.last_stderr.push(line);
        }
    }

    fn shutdown(&self) {
        self.stop_flag.store(true, Ordering::SeqCst);
        #[cfg(not(target_os = "android"))]
        {
            let Ok(mut rt) = self.runtime.lock() else {
                return;
            };
            if let Some(mut child) = rt.child.take() {
                let _ = child.kill();
                let _ = child.wait();
            }
            #[cfg(windows)]
            {
                // Dropping JobObject closes its Win32 handle, which terminates any remaining
                // descendant processes via JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE.
                rt.job.take();
            }
        }
    }
}

impl Drop for SidecarController {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(windows)]
mod win_job {
    use std::ffi::c_void;
    use std::mem::size_of;
    use std::os::windows::io::AsRawHandle;
    use std::process::Child;
    use std::ptr::null_mut;

    pub const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    const JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS: i32 = 9;
    const JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE: u32 = 0x0000_2000;

    #[repr(C)]
    #[derive(Default)]
    struct JobObjectBasicLimitInformation {
        per_process_user_time_limit: i64,
        per_job_user_time_limit: i64,
        limit_flags: u32,
        minimum_working_set_size: usize,
        maximum_working_set_size: usize,
        active_process_limit: u32,
        affinity: usize,
        priority_class: u32,
        scheduling_class: u32,
    }

    #[repr(C)]
    #[derive(Default)]
    struct IoCounters {
        read_operation_count: u64,
        write_operation_count: u64,
        other_operation_count: u64,
        read_transfer_count: u64,
        write_transfer_count: u64,
        other_transfer_count: u64,
    }

    #[repr(C)]
    #[derive(Default)]
    pub struct JobObjectExtendedLimitInformation {
        basic_limit_information: JobObjectBasicLimitInformation,
        io_info: IoCounters,
        process_memory_limit: usize,
        job_memory_limit: usize,
        peak_process_memory_used: usize,
        peak_job_memory_used: usize,
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn CreateJobObjectW(lp_job_attributes: *mut c_void, lp_name: *const u16) -> *mut c_void;
        fn SetInformationJobObject(
            h_job: *mut c_void,
            job_object_information_class: i32,
            lp_job_object_information: *const c_void,
            cb_job_object_information_length: u32,
        ) -> i32;
        fn AssignProcessToJobObject(h_job: *mut c_void, h_process: *mut c_void) -> i32;
        fn CloseHandle(h_object: *mut c_void) -> i32;
    }

    pub struct JobObject {
        handle: *mut c_void,
    }

    // Win32 HANDLE is safe to send/synchronize across threads when guarded by a Mutex.
    unsafe impl Send for JobObject {}
    unsafe impl Sync for JobObject {}

    impl JobObject {
        pub fn new_kill_on_close() -> Result<Self, String> {
            let handle = unsafe { CreateJobObjectW(null_mut(), std::ptr::null()) };
            if handle.is_null() {
                return Err(format!(
                    "CreateJobObjectW failed: {}",
                    std::io::Error::last_os_error()
                ));
            }
            let job = Self { handle };
            let mut info = JobObjectExtendedLimitInformation::default();
            info.basic_limit_information.limit_flags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            let ok = unsafe {
                SetInformationJobObject(
                    job.handle,
                    JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
                    (&info as *const JobObjectExtendedLimitInformation).cast::<c_void>(),
                    size_of::<JobObjectExtendedLimitInformation>() as u32,
                )
            };
            if ok == 0 {
                return Err(format!(
                    "SetInformationJobObject failed: {}",
                    std::io::Error::last_os_error()
                ));
            }
            Ok(job)
        }

        pub fn assign(&self, child: &Child) -> Result<(), String> {
            let ok = unsafe { AssignProcessToJobObject(self.handle, child.as_raw_handle().cast()) };
            if ok == 0 {
                return Err(format!(
                    "AssignProcessToJobObject failed: {}",
                    std::io::Error::last_os_error()
                ));
            }
            Ok(())
        }
    }

    impl Drop for JobObject {
        fn drop(&mut self) {
            if !self.handle.is_null() {
                unsafe {
                    let _ = CloseHandle(self.handle);
                }
                self.handle = null_mut();
            }
        }
    }

    #[cfg(test)]
    pub fn extended_limit_info_size() -> usize {
        size_of::<JobObjectExtendedLimitInformation>()
    }
}

#[cfg(not(target_os = "android"))]
fn pick_free_port() -> Result<u16, String> {
    let listener = TcpListener::bind("127.0.0.1:0")
        .map_err(|e| format!("Bos port secilemedi: {e}"))?;
    let port = listener
        .local_addr()
        .map_err(|e| format!("Port adresi okunamadi: {e}"))?
        .port();
    Ok(port)
}

#[cfg(not(target_os = "android"))]
fn sidecar_binary_name() -> &'static str {
    if cfg!(windows) {
        "tracera-api.exe"
    } else {
        "tracera-api"
    }
}

#[cfg(not(target_os = "android"))]
fn resolve_sidecar_path(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    let bin = sidecar_binary_name();
    let mut candidates: Vec<PathBuf> = Vec::new();

    if let Ok(res_dir) = app.path().resource_dir() {
        candidates.push(res_dir.join("tracera-api").join(bin));
        candidates.push(res_dir.join("tracera-api").join("tracera-api").join(bin));
        candidates.push(res_dir.join(bin));
    }

    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            candidates.push(exe_dir.join("tracera-api").join(bin));
            candidates.push(exe_dir.join("resources").join("tracera-api").join(bin));
        }
    }

    let dev_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../dist/tracera-api")
        .join(bin);
    candidates.push(dev_path);

    for path in &candidates {
        if path.is_file() {
            return Ok(path.clone());
        }
    }

    let checked = candidates
        .iter()
        .map(|p| p.display().to_string())
        .collect::<Vec<_>>()
        .join(" | ");
    Err(format!("Sidecar bulunamadi ({bin}). Kontrol edilen yollar: {checked}"))
}

#[cfg(not(target_os = "android"))]
fn probe_health(port: u16) -> bool {
    let addr: SocketAddr = ([127, 0, 0, 1], port).into();
    let Ok(mut stream) = TcpStream::connect_timeout(&addr, Duration::from_millis(400)) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_millis(500)));
    let _ = stream.set_write_timeout(Some(Duration::from_millis(400)));

    let req = format!(
        "GET /health HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n"
    );
    if stream.write_all(req.as_bytes()).is_err() {
        return false;
    }

    let mut buf = [0u8; 256];
    let Ok(n) = stream.read(&mut buf) else {
        return false;
    };
    if n == 0 {
        return false;
    }
    let head = String::from_utf8_lossy(&buf[..n]);
    head.starts_with("HTTP/1.1 200") || head.starts_with("HTTP/1.0 200")
}

#[cfg(not(target_os = "android"))]
fn spawn_and_monitor_sidecar(
    app: &tauri::AppHandle,
    controller: Arc<SidecarController>,
    launch_started: Instant,
) -> Result<(), String> {
    let exe_path = resolve_sidecar_path(app)?;
    let work_dir = exe_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));

    #[cfg(windows)]
    let job = win_job::JobObject::new_kill_on_close()?;

    let mut cmd = Command::new(&exe_path);
    cmd.args(["--host", "127.0.0.1", "--port", &controller.port.to_string()])
        .current_dir(&work_dir)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(win_job::CREATE_NO_WINDOW);
    }

    let mut child = cmd
        .spawn()
        .map_err(|e| format!("Sidecar baslatilamadi ({}): {e}", exe_path.display()))?;

    #[cfg(windows)]
    if let Err(err) = job.assign(&child) {
        let _ = child.kill();
        let _ = child.wait();
        return Err(err);
    }

    if let Some(stderr) = child.stderr.take() {
        let ctrl_stderr = Arc::clone(&controller);
        std::thread::spawn(move || {
            let reader = BufReader::new(stderr);
            for line in reader.lines().map_while(Result::ok) {
                ctrl_stderr.push_stderr_line(line);
            }
        });
    }

    {
        let mut rt = controller.runtime.lock().expect("sidecar runtime lock poisoned");
        rt.child = Some(child);
        #[cfg(windows)]
        {
            rt.job = Some(job);
        }
    }

    let app_handle = app.clone();
    let ctrl_poll = Arc::clone(&controller);
    std::thread::spawn(move || {
        let timeout = Duration::from_secs(120);
        while !ctrl_poll.stop_flag.load(Ordering::SeqCst) {
            if probe_health(ctrl_poll.port) {
                let startup_ms = launch_started.elapsed().as_millis() as u64;
                {
                    let mut rt = ctrl_poll.runtime.lock().expect("sidecar runtime lock poisoned");
                    rt.ready = true;
                    rt.startup_ms = Some(startup_ms);
                    rt.error = None;
                }
                let payload = ApiReadyPayload {
                    port: ctrl_poll.port,
                    base_url: ctrl_poll.base_url.clone(),
                    startup_ms,
                };
                let _ = app_handle.emit("tracera://api-ready", payload);
                return;
            }

            {
                let mut rt = ctrl_poll.runtime.lock().expect("sidecar runtime lock poisoned");
                if let Some(child) = rt.child.as_mut() {
                    if let Ok(Some(exit_status)) = child.try_wait() {
                        let tail = rt.last_stderr.join("\n");
                        rt.error = Some(format!(
                            "Sidecar erken kapandi ({exit_status}). {tail}"
                        ));
                        return;
                    }
                }
            }

            if launch_started.elapsed() >= timeout {
                let mut rt = ctrl_poll.runtime.lock().expect("sidecar runtime lock poisoned");
                rt.error = Some("Sidecar /health 120 sn icinde hazir olmadi.".to_string());
                return;
            }

            std::thread::sleep(Duration::from_millis(100));
        }
    });

    Ok(())
}

#[tauri::command]
fn get_api_port(controller: tauri::State<'_, Arc<SidecarController>>) -> u16 {
    controller.port
}

#[tauri::command]
fn get_api_base_url(controller: tauri::State<'_, Arc<SidecarController>>) -> String {
    controller.base_url.clone()
}

#[tauri::command]
fn get_sidecar_status(controller: tauri::State<'_, Arc<SidecarController>>) -> SidecarStatusDto {
    controller.status()
}

#[tauri::command]
fn get_platform() -> &'static str {
    if cfg!(target_os = "windows") {
        "windows"
    } else if cfg!(target_os = "android") {
        "android"
    } else {
        "other"
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    #[cfg(not(target_os = "android"))]
    let launch_started = Instant::now();
    #[cfg(not(target_os = "android"))]
    let (port, available) = (pick_free_port().expect("Bos yerel port secilemedi"), true);
    #[cfg(target_os = "android")]
    let (port, available) = (0u16, false);

    let controller = Arc::new(SidecarController::new(port, available));
    #[cfg(not(target_os = "android"))]
    let controller_setup = Arc::clone(&controller);

    let app = tauri::Builder::default()
        .manage(Arc::clone(&controller))
        .invoke_handler(tauri::generate_handler![
            get_api_port,
            get_api_base_url,
            get_sidecar_status,
            get_platform
        ])
        .setup(move |_app| {
            #[cfg(not(target_os = "android"))]
            if let Err(err) =
                spawn_and_monitor_sidecar(_app.handle(), Arc::clone(&controller_setup), launch_started)
            {
                let mut rt = controller_setup
                    .runtime
                    .lock()
                    .expect("sidecar runtime lock poisoned");
                rt.error = Some(err);
            }
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("Tracera uygulamasi baslatilamadi");

    app.run(move |_app_handle, event| {
        if matches!(event, RunEvent::ExitRequested { .. } | RunEvent::Exit) {
            controller.shutdown();
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(target_os = "android"))]
    #[test]
    fn picks_valid_ephemeral_port() {
        let port = pick_free_port().expect("should bind ephemeral port");
        assert!(port > 0);
    }

    #[cfg(windows)]
    #[test]
    fn win32_job_object_struct_and_creation() {
        assert_eq!(win_job::extended_limit_info_size(), 144);
        let _job = win_job::JobObject::new_kill_on_close()
            .expect("SetInformationJobObject should accept JOBOBJECT_EXTENDED_LIMIT_INFORMATION");
    }

    #[test]
    fn contract_v5_v6_ipc_payload_keys_and_platform() {
        let ready_json = serde_json::to_value(ApiReadyPayload {
            port: 53817,
            base_url: "http://127.0.0.1:53817".to_string(),
            startup_ms: 1420,
        })
        .expect("serialize ApiReadyPayload");
        assert_eq!(ready_json["port"], 53817);
        assert_eq!(ready_json["base_url"], "http://127.0.0.1:53817");
        assert_eq!(ready_json["startup_ms"], 1420);
        assert!(ready_json.get("baseUrl").is_none());

        let status_json = serde_json::to_value(SidecarStatusDto {
            available: false,
            port: 0,
            base_url: String::new(),
            ready: false,
            startup_ms: None,
            error: None,
        })
        .expect("serialize SidecarStatusDto");
        assert_eq!(status_json["available"], false);
        assert_eq!(status_json["base_url"], "");
        assert!(status_json["startup_ms"].is_null());
        assert!(status_json.get("baseUrl").is_none());

        let desktop_ctrl = SidecarController::new(53817, true);
        let desktop_status = desktop_ctrl.status();
        assert!(desktop_status.available);
        assert_eq!(desktop_status.base_url, "http://127.0.0.1:53817");

        let android_ctrl = SidecarController::new(0, false);
        let android_status = android_ctrl.status();
        assert!(!android_status.available);
        assert!(!android_status.ready);
        assert!(android_status.error.is_none());
        assert_eq!(android_status.base_url, "");

        #[cfg(windows)]
        assert_eq!(get_platform(), "windows");
        #[cfg(target_os = "android")]
        assert_eq!(get_platform(), "android");
    }
}
