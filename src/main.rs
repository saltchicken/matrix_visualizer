use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use macroquad::prelude::*;
use rustfft::{num_complex::Complex, FftPlanner};
use std::sync::atomic::{AtomicU32, Ordering};

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
    let fft = planner.plan_fft_forward(1024);
    let mut buffer = vec![0.0f32; 1024];
    let mut idx = 0;

    let stream = device.build_input_stream(
        config,
        move |data: &[T], _: &_| {
            for sample in data {
                buffer[idx] = sample.to_sample::<f32>();
                idx += 1;
                
                if idx >= 1024 {
                    // Convert audio buffer into complex numbers for FFT
                    let mut complex_buffer: Vec<Complex<f32>> = buffer
                        .iter()
                        .map(|&v| Complex { re: v, im: 0.0 })
                        .collect();
                    
                    fft.process(&mut complex_buffer);

                    // Analyze Frequency Bins
                    let mut bass = 0.0;
                    for i in 1..=6 { bass += complex_buffer[i].norm(); } // Low frequencies
                    
                    let mut mid = 0.0;
                    for i in 12..=45 { mid += complex_buffer[i].norm(); } // Mid frequencies

                    // Smooth values so the visual doesn't flicker violently
                    let old_bass = get_bass();
                    let new_bass = old_bass * 0.7 + (bass / 6.0) * 0.3;
                    store_bass(new_bass);

                    let old_mid = get_mid();
                    let new_mid = old_mid * 0.7 + (mid / 34.0) * 0.3;
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

    let chars: Vec<char> = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜﾝ".chars().collect();
    let font_size = 18.0;
    
    let mut columns = (screen_width() / font_size) as usize;
    let mut drops = vec![0.0f32; columns];

    loop {
        clear_background(BLACK);
        
        // Handle window resize dynamically
        let current_cols = (screen_width() / font_size) as usize;
        if current_cols != columns {
            drops.resize(current_cols, 0.0);
            columns = current_cols;
        }

        let bass = get_bass();
        let mid = get_mid();

        // 1. Reactive Color: Heavy bass pushes the green toward bright white/cyan
        let bass_overdrive = if bass > 12.0 { (bass - 12.0) / 15.0 } else { 0.0 };
        let r = bass_overdrive.clamp(0.0, 1.0);
        let b = bass_overdrive.clamp(0.0, 1.0);

        let tail_length = 15;

        for i in 0..columns {
            // 2. Reactive Speed: Bass spikes speed up the rain
            let speed = if bass > 15.0 && rand::gen_range(0.0, 1.0) > 0.5 { 2.0 } else { 1.0 };
            drops[i] += speed;

            // 3. Reactive Glitch: High bass resets trails at random heights
            let glitch_chance = if bass > 18.0 { 0.90 } else { 0.99 };
            if drops[i] * font_size > screen_height() && rand::gen_range(0.0, 1.0) > glitch_chance {
                drops[i] = 0.0;
            }

            // Draw the fading trail procedurally
            for t in 0..tail_length {
                let y = (drops[i] - t as f32) * font_size;
                
                if y > 0.0 {
                    let alpha = 1.0 - (t as f32 / tail_length as f32);
                    
                    // 4. Reactive Trails: Loud mids make the trails thicker/longer
                    let modified_alpha = (alpha + (mid * 0.01)).clamp(0.0, 1.0);
                    
                    let color = Color::new(r, 1.0, b, modified_alpha);
                    let text = chars[rand::gen_range(0, chars.len())].to_string();
                    
                    draw_text(&text, i as f32 * font_size, y, font_size, color);
                }
            }
        }

        next_frame().await
    }
}
