use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use macroquad::prelude::*;
use notify_debouncer_mini::{new_debouncer, notify::RecursiveMode};
use rustfft::{num_complex::Complex, FftPlanner};
use serde::Deserialize;
use std::path::Path;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{mpsc, Arc, RwLock};
use std::time::Duration;

// ==========================================
// --- CONFIGURATION STRUCTURE ---
// ==========================================

#[derive(Debug, Clone, Deserialize)]
struct Config {
    audio: AudioConfig,
    visuals: VisualConfig,
    movement: MovementConfig,
}

#[derive(Debug, Clone, Deserialize)]
struct AudioConfig {
    bass_bin_start: usize,
    bass_bin_end: usize,
    mid_bin_start: usize,
    mid_bin_end: usize,
    smoothing_old: f32,
    smoothing_new: f32,
    bass_norm_factor: f32,
    mid_norm_factor: f32,
}

#[derive(Debug, Clone, Deserialize)]
struct VisualConfig {
    font_size: f32,
    tail_length: usize,
    drop_min_start_y: f32,
    drop_max_start_y: f32,
    drop_min_reset_y: f32,
    drop_max_reset_y: f32,
}

#[derive(Debug, Clone, Deserialize)]
struct MovementConfig {
    fall_speed_base: f32,
    fall_speed_beat_pump: f32,
    beat_decay: f32,
    beat_threshold_multi: f32,
    beat_threshold_min: f32,
    base_char_changes_per_sec: f64,
}

// Hardcoded structural constraints (not hot-reloaded)
const AUDIO_BUFFER_SIZE: usize = 1024;
const HASH_PRIME_1: u64 = 137;
const HASH_PRIME_2: u64 = 251;

// ==========================================

static BASS_VAL: AtomicU32 = AtomicU32::new(0);
static MID_VAL: AtomicU32 = AtomicU32::new(0);

fn store_bass(val: f32) { BASS_VAL.store(val.to_bits(), Ordering::Relaxed); }
fn get_bass() -> f32 { f32::from_bits(BASS_VAL.load(Ordering::Relaxed)) }

fn store_mid(val: f32) { MID_VAL.store(val.to_bits(), Ordering::Relaxed); }
fn get_mid() -> f32 { f32::from_bits(MID_VAL.load(Ordering::Relaxed)) }

// Helper to load config from disk
fn load_config() -> Config {
    let contents = std::fs::read_to_string("config.toml")
        .expect("Failed to read config.toml");
    toml::from_str(&contents).expect("Failed to parse config.toml")
}

// --- Audio Thread --- //
fn build_audio_stream(shared_config: Arc<RwLock<Config>>) -> Result<cpal::Stream, anyhow::Error> {
    let host = cpal::default_host();
    let device = host.default_input_device().ok_or_else(|| anyhow::anyhow!("No input device found"))?;
    let config = device.default_input_config()?;
    
    println!("Audio Client initialized. Look for it in qpwgraph!");

    let stream = match config.sample_format() {
        cpal::SampleFormat::F32 => run_stream::<f32>(&device, &config.into(), shared_config),
        cpal::SampleFormat::I16 => run_stream::<i16>(&device, &config.into(), shared_config),
        _ => anyhow::bail!("Unsupported sample format"),
    }?;
    
    Ok(stream)
}

fn run_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    shared_config: Arc<RwLock<Config>>,
) -> Result<cpal::Stream, anyhow::Error>
where
    T: cpal::Sample + cpal::SizedSample,
    f32: cpal::FromSample<T>,
{
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(AUDIO_BUFFER_SIZE);
    let mut buffer = vec![0.0f32; AUDIO_BUFFER_SIZE];
    let mut idx = 0;

    let stream = device.build_input_stream(
        config,
        move |data: &[T], _: &_| {
            // Grab a quick copy of the audio config settings for this processing tick
            let cfg = shared_config.read().unwrap().audio.clone();

            for sample in data {
                buffer[idx] = sample.to_sample::<f32>();
                idx += 1;
                
                if idx >= AUDIO_BUFFER_SIZE {
                    let mut complex_buffer: Vec<Complex<f32>> = buffer
                        .iter()
                        .map(|&v| Complex { re: v, im: 0.0 })
                        .collect();
                    
                    fft.process(&mut complex_buffer);

                    let mut bass = 0.0;
                    for i in cfg.bass_bin_start..=cfg.bass_bin_end { bass += complex_buffer[i].norm(); }
                    
                    let mut mid = 0.0;
                    for i in cfg.mid_bin_start..=cfg.mid_bin_end { mid += complex_buffer[i].norm(); }

                    let old_bass = get_bass();
                    let new_bass = old_bass * cfg.smoothing_old + (bass / cfg.bass_norm_factor) * cfg.smoothing_new;
                    store_bass(new_bass);

                    let old_mid = get_mid();
                    let new_mid = old_mid * cfg.smoothing_old + (mid / cfg.mid_norm_factor) * cfg.smoothing_new;
                    store_mid(new_mid);

                    idx = 0;
                }
            }
        },
        |err| eprintln!("Audio stream error: {}", err),
        None,
    )?;
    
    stream.play()?;
    Ok(stream)
}

// --- Graphics Thread --- //
#[macroquad::main("Reactive Matrix")]
async fn main() {
    // 1. Initialize Configuration
    let shared_config = Arc::new(RwLock::new(load_config()));

    // 2. Setup notify-debouncer-mini
    let (tx, rx) = mpsc::channel();
    
    // Create the debouncer with a 250ms timeout window
    let mut debouncer = new_debouncer(Duration::from_millis(250), tx)
        .expect("Failed to create file watcher");

    // Watch the current directory
    debouncer.watcher().watch(Path::new("."), RecursiveMode::NonRecursive)
           .expect("Failed to watch directory");

    // 3. Initialize Audio & Assets
    let _audio_stream = build_audio_stream(Arc::clone(&shared_config))
        .expect("Failed to start audio stream");

    let mut matrix_font = load_ttf_font("assets/SpaceMono-Regular.ttf")
        .await
        .expect("Failed to load font. Ensure SpaceMono-Regular.ttf is in assets/.");
    matrix_font.set_filter(FilterMode::Nearest);

    let chars: Vec<char> = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ".chars().collect();
    
    let initial_font_size = shared_config.read().unwrap().visuals.font_size;
    let mut columns = (screen_width() / initial_font_size) as usize;
    
    let mut drops: Vec<f32> = (0..columns).map(|_| rand::gen_range(-50.0, 0.0)).collect();
    let mut drop_speeds: Vec<f32> = (0..columns).map(|_| rand::gen_range(0.75, 1.25)).collect();

    let mut bass_avg = 0.0;
    let mut beat_envelope = 0.0f32;

    loop {
        clear_background(BLACK);
        
        // --- DEBOUNCER LOGIC ---
        // notify-debouncer-mini handles the timers. We just check if a clean, batched 
        // event group arrived for config.toml.
        let mut reload_config = false;
        while let Ok(res) = rx.try_recv() {
            if let Ok(events) = res {
                // If any event in the batch involves config.toml, mark for reload
                if events.iter().any(|e| e.path.file_name().is_some_and(|name| name == "config.toml")) {
                    reload_config = true;
                }
            }
        }
        
        if reload_config {
            if let Ok(contents) = std::fs::read_to_string("config.toml") {
                match toml::from_str::<Config>(&contents) {
                    Ok(new_config) => {
                        *shared_config.write().unwrap() = new_config;
                        println!("Config hot-reloaded successfully!");
                    }
                    Err(e) => eprintln!("Config parse error (ignoring): {}", e),
                }
            }
        }
        // -----------------------

        // Lock config once per frame for rendering
        let cfg = shared_config.read().unwrap().clone();
        
        let current_cols = (screen_width() / cfg.visuals.font_size) as usize;
        if current_cols != columns {
            drops.resize_with(current_cols, || rand::gen_range(cfg.visuals.drop_min_start_y, cfg.visuals.drop_max_start_y));
            drop_speeds.resize_with(current_cols, || rand::gen_range(0.75, 1.25));
            columns = current_cols;
        }

        let bass = get_bass();
        let mid = get_mid();

        if bass > (bass_avg * cfg.movement.beat_threshold_multi) + cfg.movement.beat_threshold_min {
            beat_envelope = 1.0;
        }
        
        bass_avg = bass_avg * 0.95 + bass * 0.05;
        beat_envelope *= cfg.movement.beat_decay; 
        
        let scramble_speed = cfg.movement.base_char_changes_per_sec + (mid as f64 * 0.4) + (beat_envelope as f64 * 25.0);
        let time_step = (get_time() * scramble_speed) as u64;

        for i in 0..columns {
            let speed = (cfg.movement.fall_speed_base + (cfg.movement.fall_speed_beat_pump * beat_envelope)) * drop_speeds[i];
            drops[i] += speed;

            if (drops[i] - cfg.visuals.tail_length as f32) * cfg.visuals.font_size > screen_height() {
                drops[i] = rand::gen_range(cfg.visuals.drop_min_reset_y, cfg.visuals.drop_max_reset_y);
            }

            let head_idx = drops[i].floor();
            let x = i as f32 * cfg.visuals.font_size;

            for t in 0..cfg.visuals.tail_length {
                let y_idx = head_idx - t as f32;
                let y = y_idx * cfg.visuals.font_size;
                
                if y > -cfg.visuals.font_size && y < screen_height() + cfg.visuals.font_size {
                    let base_alpha = 1.0 - (t as f32 / cfg.visuals.tail_length as f32);
                    let alpha = (base_alpha + (mid * 0.01)).clamp(0.0, 1.0);
                    
                    let color = if t == 0 {
                        Color::new(1.0, 1.0, 1.0, 1.0)
                    } else {
                        let r_pulse = 0.0 + (0.5 * beat_envelope);
                        let g_pulse = 0.8 + (0.2 * beat_envelope);
                        let b_pulse = 0.2 + (0.8 * beat_envelope);
                        Color::new(r_pulse, g_pulse, b_pulse, alpha)
                    };
                    
                    let hash = (i as u64)
                        .wrapping_mul(HASH_PRIME_1)
                        .wrapping_add((y_idx as i64) as u64)
                        .wrapping_mul(HASH_PRIME_2)
                        .wrapping_add(time_step);
                        
                    let char_idx = (hash % chars.len() as u64) as usize;
                    let text = chars[char_idx].to_string();
                    
                    let glitch_offset = beat_envelope * 6.0; 
                    
                    if glitch_offset > 0.5 && t < 3 {
                        draw_text_ex(&text, x + glitch_offset, y, TextParams {
                            font: Some(&matrix_font),
                            font_size: cfg.visuals.font_size as u16,
                            color: Color::new(0.0, 1.0, 1.0, alpha * beat_envelope),
                            ..Default::default()
                        });
                        
                        draw_text_ex(&text, x - glitch_offset, y, TextParams {
                            font: Some(&matrix_font),
                            font_size: cfg.visuals.font_size as u16,
                            color: Color::new(1.0, 0.0, 0.5, alpha * beat_envelope),
                            ..Default::default()
                        });
                    }

                    draw_text_ex(&text, x, y, TextParams {
                        font: Some(&matrix_font),
                        font_size: cfg.visuals.font_size as u16,
                        color,
                        ..Default::default()
                    });
                }
            }
        }

        next_frame().await
    }
}
