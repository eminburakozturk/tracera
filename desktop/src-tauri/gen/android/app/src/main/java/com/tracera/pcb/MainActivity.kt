package com.tracera.pcb

import android.annotation.SuppressLint
import android.os.Bundle
import android.webkit.WebView
import androidx.activity.enableEdgeToEdge

class MainActivity : TauriActivity() {
  override fun onCreate(savedInstanceState: Bundle?) {
    enableEdgeToEdge()
    super.onCreate(savedInstanceState)
  }

  @SuppressLint("SetJavaScriptEnabled")
  override fun onWebViewCreate(webView: WebView) {
    super.onWebViewCreate(webView)
    // Allow getUserMedia <video autoPlay> stream playback without an extra media gesture.
    webView.settings.mediaPlaybackRequiresUserGesture = false
    webView.settings.domStorageEnabled = true
  }
}
