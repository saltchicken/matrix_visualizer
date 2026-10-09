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

// Speed & Movement Settings
const FALL_SPEED_BASE: f32 = 0.25;
const FALL_SPEED_REACTIVE: f32 = 0.8;
const REACTIVE_SPEED_BASS_THRESHOLD: f32 = 15.0;
const REACTIVE_SPEED_CHANCE: f32 = 0.5;
const CHAR_CHANGES_PER_SECOND: f64 = 5.0; // How many times per second the characters flip

// Audio Reactivity Adjustments
const BASS_OVERDRIVE_THRESHOLD: f32 = 12.0;
const BASS_OVERDRIVE_DIVISOR: f32 = 15.0;
const MID_ALPHA_MULTIPLIER: f32 = 0.01;

// Colors
const HEAD_COLOR_R: f32 = 0.9;
const HEAD_COLOR_G: f32 = 1.0;
const HEAD_COLOR_B: f32 = 0.9;
const HEAD_COLOR_A: f32 = 1.0;

// Math Primes (Used for character randomization hashing)
const HASH_PRIME_1: u64 = 137;
const HASH_PRIME_2: u64 = 251;

// ==========================================

// Shared atomic variables for audio-visual synchronization
static BASS_VAL: AtomicU32 = AtomicU32::new(0);
static MID_VAL: AtomicU32 = AtomicU32::new(0);

// Helper functions to safely pass f32 across threads
fn store_bass(val: f32) { BASS_VAL.store(val.to_bits(), Ordering::Relaxed); }
fn get_bass() -> f32 { f32::from_bits(BASS_VAL.load(Ordering::Relaxed)) }

fn store_mid(val: f32) { MID_VAL.store(val.to_bits(), Ordering::Relaxed); }
fn get_mid() -> f32 { f32::from_bits(MID_VAL.load(Ordering::Relaxed)) }

// --- Audio Thread --- //
fn build_audio_stream() -> Result<cpal::Stream, anyhow::Error> {
    let host = cpal::default_host();
    
    // Grabs the default input device. In PipeWire, this creates a recordable client.
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
                    // Convert audio buffer into complex numbers for FFT
                    let mut complex_buffer: Vec<Complex<f32>> = buffer
                        .iter()
                        .map(|&v| Complex { re: v, im: 0.0 })
                        .collect();
                    
                    fft.process(&mut complex_buffer);

                    // Analyze Frequency Bins
                    let mut bass = 0.0;
                    for i in BASS_BIN_START..=BASS_BIN_END { bass += complex_buffer[i].norm(); } // Low frequencies
                    
                    let mut mid = 0.0;
                    for i in MID_BIN_START..=MID_BIN_END { mid += complex_buffer[i].norm(); } // Mid frequencies

                    // Smooth values so the visual doesn't flicker violently
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

    // 1. Load the custom Space Mono font
    let mut matrix_font = load_ttf_font("assets/SpaceMono-Regular.ttf")
        .await
        .expect("Failed to load font. Ensure SpaceMono-Regular.ttf is in the assets/ directory.");

    // Force integer-ratio nearest-neighbor scaling so the text doesn't blur
    matrix_font.set_filter(FilterMode::Nearest);

    // Swapped to ASCII since Space Mono does not support Japanese Katakana
    let chars: Vec<char> = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ".chars().collect();
    
    let mut columns = (screen_width() / FONT_SIZE) as usize;
    
    // STAGGER INITIALIZATION: Randomize start positions so the screen is filled immediately
    let mut drops: Vec<f32> = (0..columns)
        .map(|_| rand::gen_range(DROP_MIN_START_Y, screen_height() / FONT_SIZE))
        .collect();

    loop {
        clear_background(BLACK);
        
        // Handle window resize dynamically
        let current_cols = (screen_width() / FONT_SIZE) as usize;
        if current_cols != columns {
            // Fill new columns with staggered heights above the screen
            drops.resize_with(current_cols, || rand::gen_range(DROP_MIN_START_Y, DROP_MAX_START_Y));
            columns = current_cols;
        }

        let bass = get_bass();
        let mid = get_mid();

        let bass_overdrive = if bass > BASS_OVERDRIVE_THRESHOLD { 
            (bass - BASS_OVERDRIVE_THRESHOLD) / BASS_OVERDRIVE_DIVISOR 
        } else { 
            0.0 
        };
        
        let r = bass_overdrive.clamp(0.0, 1.0);
        let b = bass_overdrive.clamp(0.0, 1.0);

        for i in 0..columns {
            // DYNAMIC SPEED: Drops progress fractionally per frame
            let speed = if bass > REACTIVE_SPEED_BASS_THRESHOLD && rand::gen_range(0.0, 1.0) > REACTIVE_SPEED_CHANCE { 
                FALL_SPEED_REACTIVE 
            } else { 
                FALL_SPEED_BASE 
            };
            drops[i] += speed;

            // CONTINUOUS WRAP: Reset drops to a random negative position above the screen to prevent gaps
            if (drops[i] - TAIL_LENGTH as f32) * FONT_SIZE > screen_height() {
                drops[i] = rand::gen_range(DROP_MIN_RESET_Y, DROP_MAX_RESET_Y);
            }

            // GRID SNAPPING: Floor the float to make characters "tick" downwards instead of sliding
            let head_idx = drops[i].floor();

            for t in 0..TAIL_LENGTH {
                let y_idx = head_idx - t as f32;
                let y = y_idx * FONT_SIZE;
                
                // Only draw characters that are currently on screen
                if y > -FONT_SIZE && y < screen_height() + FONT_SIZE {
                    let alpha = 1.0 - (t as f32 / TAIL_LENGTH as f32);
                    let modified_alpha = (alpha + (mid * MID_ALPHA_MULTIPLIER)).clamp(0.0, 1.0);
                    
                    // BRIGHT HEAD: The leading character is bright white/green, trailing is green
                    let color = if t == 0 {
                        Color::new(HEAD_COLOR_R, HEAD_COLOR_G, HEAD_COLOR_B, HEAD_COLOR_A)
                    } else {
                        Color::new(r, 1.0, b, modified_alpha)
                    };
                    
                    // TIME-BASED CHARACTER HASHING
                    // Control how fast the characters flip instead of changing every frame
                    let time_step = (get_time() * CHAR_CHANGES_PER_SECOND) as u64;

                    // Create a predictable hash based on the column (i), row (y_idx), and time
                    let hash = (i as u64)
                        .wrapping_mul(HASH_PRIME_1)
                        .wrapping_add((y_idx as i64) as u64)
                        .wrapping_mul(HASH_PRIME_2)
                        .wrapping_add(time_step);
                        
                    let char_idx = (hash % chars.len() as u64) as usize;
                    let text = chars[char_idx].to_string();
                    
                    // 2. Use draw_text_ex to apply the custom Space Mono font
                    draw_text_ex(
                        &text,
                        i as f32 * FONT_SIZE,
                        y,
                        TextParams {
                            font: Some(&matrix_font),
                            font_size: FONT_SIZE as u16,
                            color,
                            ..Default::default()
                        },
                    );
                }
            }
        }

        next_frame().await
    }
}
