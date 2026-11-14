/**
 * Örnek veri (Mock) motoru yalnız geliştirme/test ortamında görünür.
 * Üretim derlemesinde (web, EXE, APK) seçici ve rozet çıkmaz.
 * E2E testleri `npm run dev` ile çalıştığı için DEV=true olur.
 */
export const MOCK_ENABLED: boolean =
  import.meta.env.DEV ||
  import.meta.env.MODE === "test" ||
  import.meta.env.VITE_SHOW_MOCK === "1";
