use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use macroquad::prelude::*;
use rustfft::{num_complex::Complex, FftPlanner};
use std::sync::atomic::{AtomicU32, Ordering};

// ==========================================
// --- CONFIGURATION / MAGIC NUMBERS ---
// ==========================================

// Audio & FFT Settings
const AUDIO_BUFFER_SIZE: usize = 1024;
const BASS_BIN_START: usize = 1;
const BASS_BIN_END: usize = 6;
const MID_BIN_START: usize = 12;
const MID_BIN_END: usize = 45;
const SMOOTHING_OLD: f32 = 0.7;
const SMOOTHING_NEW: f32 = 0.3;
const BASS_NORM_FACTOR: f32 = 6.0;
const MID_NORM_FACTOR: f32 = 34.0;

// Visual Layout Settings
const FONT_SIZE: f32 = 18.0;
const TAIL_LENGTH: usize = 20;
const DROP_MIN_START_Y: f32 = -50.0;
const DROP_MAX_START_Y: f32 = 0.0;
const DROP_MIN_RESET_Y: f32 = -30.0;
const DROP_MAX_RESET_Y: f32 = 0.0;

// --- REIMAGINED BEAT & MOVEMENT SETTINGS ---
const FALL_SPEED_BASE: f32 = 0.15;
const FALL_SPEED_BEAT_PUMP: f32 = 0.4; // Speed added to drops during a beat
const BEAT_DECAY: f32 = 0.85; // How smoothly the beat effect falls off (0.0 to 1.0)
const BEAT_THRESHOLD_MULTI: f32 = 1.35; // How much higher than average bass needs to be to trigger
const BEAT_THRESHOLD_MIN: f32 = 2.0; // Minimum bass required to trigger a beat
const BASE_CHAR_CHANGES_PER_SEC: f64 = 2.0;

// Math Primes (Used for character randomization hashing)
const HASH_PRIME_1: u64 = 137;
const HASH_PRIME_2: u64 = 251;

// ==========================================

static BASS_VAL: AtomicU32 = AtomicU32::new(0);
static MID_VAL: AtomicU32 = AtomicU32::new(0);

fn store_bass(val: f32) { BASS_VAL.store(val.to_bits(), Ordering::Relaxed); }
fn get_bass() -> f32 { f32::from_bits(BASS_VAL.load(Ordering::Relaxed)) }

fn store_mid(val: f32) { MID_VAL.store(val.to_bits(), Ordering::Relaxed); }
fn get_mid() -> f32 { f32::from_bits(MID_VAL.load(Ordering::Relaxed)) }

// --- Audio Thread --- //
fn build_audio_stream() -> Result<cpal::Stream, anyhow::Error> {
    let host = cpal::default_host();
    let device = host.default_input_device().ok_or_else(|| anyhow::anyhow!("No input device found"))?;
    let config = device.default_input_config()?;
    
    println!("Audio Client initialized. Look for it in qpwgraph!");

    let stream = match config.sample_format() {
        cpal::SampleFormat::F32 => run_stream::<f32>(&device, &config.into()),
        cpal::SampleFormat::I16 => run_stream::<i16>(&device, &config.into()),
        _ => anyhow::bail!("Unsupported sample format"),
    }?;
    
    Ok(stream)
}

fn run_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
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
                    for i in BASS_BIN_START..=BASS_BIN_END { bass += complex_buffer[i].norm(); }
                    
                    let mut mid = 0.0;
                    for i in MID_BIN_START..=MID_BIN_END { mid += complex_buffer[i].norm(); }

                    let old_bass = get_bass();
                    let new_bass = old_bass * SMOOTHING_OLD + (bass / BASS_NORM_FACTOR) * SMOOTHING_NEW;
                    store_bass(new_bass);

                    let old_mid = get_mid();
                    let new_mid = old_mid * SMOOTHING_OLD + (mid / MID_NORM_FACTOR) * SMOOTHING_NEW;
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
    let _audio_stream = build_audio_stream().expect("Failed to start audio stream");

    let mut matrix_font = load_ttf_font("assets/SpaceMono-Regular.ttf")
        .await
        .expect("Failed to load font. Ensure SpaceMono-Regular.ttf is in the assets/.");
    matrix_font.set_filter(FilterMode::Nearest);

    let chars: Vec<char> = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ".chars().collect();
    
    let mut columns = (screen_width() / FONT_SIZE) as usize;
    
    // Arrays for column states
    let mut drops: Vec<f32> = (0..columns).map(|_| rand::gen_range(DROP_MIN_START_Y, screen_height() / FONT_SIZE)).collect();
    let mut drop_speeds: Vec<f32> = (0..columns).map(|_| rand::gen_range(0.75, 1.25)).collect();

    // Beat Tracking Variables
    let mut bass_avg = 0.0;
    let mut beat_envelope = 0.0f32;

    loop {
        clear_background(BLACK);
        
        let current_cols = (screen_width() / FONT_SIZE) as usize;
        if current_cols != columns {
            drops.resize_with(current_cols, || rand::gen_range(DROP_MIN_START_Y, DROP_MAX_START_Y));
            drop_speeds.resize_with(current_cols, || rand::gen_range(0.75, 1.25));
            columns = current_cols;
        }

        let bass = get_bass();
        let mid = get_mid();

        // ----------------------------------------------------
        // TRANSIENT BEAT DETECTION
        // ----------------------------------------------------
        // Trigger a beat if current bass sharply exceeds the rolling average
        if bass > (bass_avg * BEAT_THRESHOLD_MULTI) + BEAT_THRESHOLD_MIN {
            beat_envelope = 1.0; // Max out the envelope!
        }
        
        // Update rolling average slowly, decay the envelope quickly
        bass_avg = bass_avg * 0.95 + bass * 0.05;
        beat_envelope *= BEAT_DECAY; 
        
        // Smoothly bring up chaotic mutations on beat + mid freq presence
        let scramble_speed = BASE_CHAR_CHANGES_PER_SEC + (mid as f64 * 0.4) + (beat_envelope as f64 * 25.0);
        let time_step = (get_time() * scramble_speed) as u64;

        for i in 0..columns {
            // Speed smoothly "pumps" with the beat envelope, scaled by the column's individual depth speed
            let speed = (FALL_SPEED_BASE + (FALL_SPEED_BEAT_PUMP * beat_envelope)) * drop_speeds[i];
            drops[i] += speed;

            if (drops[i] - TAIL_LENGTH as f32) * FONT_SIZE > screen_height() {
                drops[i] = rand::gen_range(DROP_MIN_RESET_Y, DROP_MAX_RESET_Y);
            }

            let head_idx = drops[i].floor();
            let x = i as f32 * FONT_SIZE;

            for t in 0..TAIL_LENGTH {
                let y_idx = head_idx - t as f32;
                let y = y_idx * FONT_SIZE;
                
                if y > -FONT_SIZE && y < screen_height() + FONT_SIZE {
                    let base_alpha = 1.0 - (t as f32 / TAIL_LENGTH as f32);
                    let alpha = (base_alpha + (mid * 0.01)).clamp(0.0, 1.0);
                    
                    // Colors transition from Matrix Green -> Cyberpunk Cyan/White on the beat
                    let color = if t == 0 {
                        // Bright leading head flashes brilliant white on beat
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
                    
                    // ----------------------------------------------------
                    // CHROMATIC ABERRATION GLITCH (Beat Driven)
                    // ----------------------------------------------------
                    // Only glitch the first few characters of a tail for emphasis
                    let glitch_offset = beat_envelope * 6.0; 
                    
                    if glitch_offset > 0.5 && t < 3 {
                        // Cyan offset (Right)
                        draw_text_ex(&text, x + glitch_offset, y, TextParams {
                            font: Some(&matrix_font),
                            font_size: FONT_SIZE as u16,
                            color: Color::new(0.0, 1.0, 1.0, alpha * beat_envelope),
                            ..Default::default()
                        });
                        
                        // Magenta offset (Left)
                        draw_text_ex(&text, x - glitch_offset, y, TextParams {
                            font: Some(&matrix_font),
                            font_size: FONT_SIZE as u16,
                            color: Color::new(1.0, 0.0, 0.5, alpha * beat_envelope),
                            ..Default::default()
                        });
                    }

                    // Main Text draw
                    draw_text_ex(&text, x, y, TextParams {
                        font: Some(&matrix_font),
                        font_size: FONT_SIZE as u16,
                        color,
                        ..Default::default()
                    });
                }
            }
        }

        next_frame().await
    }
}
