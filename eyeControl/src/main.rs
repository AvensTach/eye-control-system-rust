use pyo3::prelude::*;
use pyo3::types::PyList;

use opencv::{
    core::{self, Point, Scalar},
    highgui, imgproc,
    prelude::*,
    videoio,
};
use enigo::{Enigo, Mouse, Settings};
use std::fs::File;
use std::io::copy;
use std::path::Path;

const MODEL_PATH: &str = "face_landmarker.task";
const MODEL_URL: &str = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task";

const RIGHT_EYE_LEFT: usize = 362;
const RIGHT_EYE_RIGHT: usize = 263;
const RIGHT_EYE_TOP: usize = 386;
const RIGHT_EYE_BOTTOM: usize = 374;
const RIGHT_IRIS: [usize; 4] = [474, 475, 476, 477];

const LEFT_EYE_LEFT: usize = 33;
const LEFT_EYE_RIGHT: usize = 133;
const LEFT_EYE_TOP: usize = 159;
const LEFT_EYE_BOTTOM: usize = 145;
const LEFT_IRIS: [usize; 4] = [469, 470, 471, 472];

const NOSE_TIP: usize = 1;
const FOREHEAD: usize = 10;
const CHIN: usize = 152;
const LEFT_CHEEK: usize = 234;
const RIGHT_CHEEK: usize = 454;

#[derive(Clone, Copy, Default)]
struct Landmark {
    x: f32,
    y: f32,
}

#[derive(Clone, Copy, Default)]
struct Calibration {
    c: (f32, f32),
    tl: (f32, f32),
    tr: (f32, f32),
    bl: (f32, f32),
    br: (f32, f32),
}

fn download_model_if_needed() {
    if !Path::new(MODEL_PATH).exists() {
        let response = reqwest::blocking::get(MODEL_URL).expect("Failed to download model");
        let mut dest = File::create(MODEL_PATH).expect("Failed to create file");
        let content = response.bytes().expect("Failed to read bytes");
        copy(&mut content.as_ref(), &mut dest).expect("Failed to write to file");
    }
}

fn get_eye_ratio(
    landmarks: &[Landmark],
    left_idx: usize,
    right_idx: usize,
    top_idx: usize,
    bottom_idx: usize,
    iris_indices: &[usize; 4],
) -> Option<(f32, f32, (f32, f32), (f32, f32, f32, f32))> {
    let eye_left = landmarks[left_idx];
    let eye_right = landmarks[right_idx];

    let iris_x = iris_indices.iter().map(|&i| landmarks[i].x).sum::<f32>() / 4.0;
    let iris_y = iris_indices.iter().map(|&i| landmarks[i].y).sum::<f32>() / 4.0;

    let eye_width = (eye_right.x - eye_left.x).abs();

    if eye_width > 0.0 {
        let ratio_x = (iris_x - eye_left.x).abs() / eye_width;
        let stable_center_y = (eye_left.y + eye_right.y) / 2.0;
        let ratio_y = (iris_y - stable_center_y) / eye_width;

        return Some((
            ratio_x,
            ratio_y,
            (iris_x, iris_y),
            (eye_left.x, landmarks[top_idx].y, eye_right.x, landmarks[bottom_idx].y),
        ));
    }
    None
}

fn calculate_average_gaze(
    landmarks: &[Landmark],
) -> Option<(f32, f32, [(f32, f32); 2], [(f32, f32, f32, f32); 2])> {
    let right = get_eye_ratio(landmarks, RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, &RIGHT_IRIS);
    let left = get_eye_ratio(landmarks, LEFT_EYE_LEFT, LEFT_EYE_RIGHT, LEFT_EYE_TOP, LEFT_EYE_BOTTOM, &LEFT_IRIS);

    if let (Some((rx1, ry1, r_iris, r_bounds)), Some((rx2, ry2, l_iris, l_bounds))) = (right, left) {
        let avg_x = (rx1 + rx2) / 2.0;
        let avg_y = (ry1 + ry2) / 2.0;
        Some((avg_x, avg_y, [r_iris, l_iris], [r_bounds, l_bounds]))
    } else {
        None
    }
}

fn get_head_pose(landmarks: &[Landmark]) -> (f32, f32) {
    let nose = landmarks[NOSE_TIP];
    let left = landmarks[LEFT_CHEEK];
    let right = landmarks[RIGHT_CHEEK];
    let top = landmarks[FOREHEAD];

    let face_width = (right.x - left.x).abs();

    if face_width > 0.0 {
        let yaw = (nose.x - left.x) / face_width;
        let pitch = (nose.y - top.y) / face_width;
        return (yaw, pitch);
    }
    (0.5, 0.5)
}

fn get_axis_scaled_value(current: f32, min_bound: f32, center: f32, max_bound: f32) -> f32 {
    let scaled = if current <= center {
        let rng = if (center - min_bound).abs() > f32::EPSILON { center - min_bound } else { 0.0001 };
        ((current - min_bound) / rng) * 0.5
    } else {
        let rng = if (max_bound - center).abs() > f32::EPSILON { max_bound - center } else { 0.0001 };
        0.5 + (((current - center) / rng) * 0.5)
    };
    scaled.clamp(0.0, 1.0)
}

fn get_dynamic_face_weight(yaw: f32, pitch: f32, center_yaw: f32, center_pitch: f32) -> f32 {
    let min_weight = 0.1;
    let max_weight = 0.85;

    let dist_yaw = (yaw - center_yaw).abs();
    let dist_pitch = (pitch - center_pitch).abs();
    let distance = (dist_yaw.powi(2) + dist_pitch.powi(2)).sqrt();

    let max_expected_dev = 0.08;
    let normalized_dist = (distance / max_expected_dev).min(1.0);

    min_weight + (max_weight - min_weight) * normalized_dist.powi(2)
}

fn update_mouse_position(
    ratio_x: f32, ratio_y: f32, yaw: f32, pitch: f32, screen_w: f32, screen_h: f32,
    smooth_x: f32, smooth_y: f32, calib_eyes: &Calibration, calib_face: &Calibration,
    smooth_factor: f32, invert_x: bool, invert_y: bool, enigo: &mut Enigo
) -> (f32, f32, f32) {
    let eye_left_bound = (calib_eyes.tl.0 + calib_eyes.bl.0) / 2.0;
    let eye_right_bound = (calib_eyes.tr.0 + calib_eyes.br.0) / 2.0;
    let eye_top_bound = (calib_eyes.tl.1 + calib_eyes.tr.1) / 2.0;
    let eye_bottom_bound = (calib_eyes.bl.1 + calib_eyes.br.1) / 2.0;

    let face_left_bound = (calib_face.tl.0 + calib_face.bl.0) / 2.0;
    let face_right_bound = (calib_face.tr.0 + calib_face.br.0) / 2.0;
    let face_top_bound = (calib_face.tl.1 + calib_face.tr.1) / 2.0;
    let face_bottom_bound = (calib_face.bl.1 + calib_face.br.1) / 2.0;

    let eye_scaled_x = get_axis_scaled_value(ratio_x, eye_left_bound, calib_eyes.c.0, eye_right_bound);
    let eye_scaled_y = get_axis_scaled_value(ratio_y, eye_top_bound, calib_eyes.c.1, eye_bottom_bound);

    let mut face_scaled_x = get_axis_scaled_value(yaw, face_left_bound, calib_face.c.0, face_right_bound);
    let mut face_scaled_y = get_axis_scaled_value(pitch, face_top_bound, calib_face.c.1, face_bottom_bound);

    if invert_x { face_scaled_x = 1.0 - face_scaled_x; }
    if invert_y { face_scaled_y = 1.0 - face_scaled_y; }

    let current_face_weight = get_dynamic_face_weight(yaw, pitch, calib_face.c.0, calib_face.c.1);

    let final_x = (eye_scaled_x * (1.0 - current_face_weight)) + (face_scaled_x * current_face_weight);
    let final_y = (eye_scaled_y * (1.0 - current_face_weight)) + (face_scaled_y * current_face_weight);

    let target_x = final_x * screen_w;
    let target_y = final_y * screen_h;

    let dead_zone_radius = 25.0;
    let dx = target_x - smooth_x;
    let dy = target_y - smooth_y;
    let distance = (dx.powi(2) + dy.powi(2)).sqrt();

    let active_smooth_factor = if distance < dead_zone_radius {
        smooth_factor * 0.1
    } else {
        smooth_factor
    };

    let new_smooth_x = smooth_x + dx * active_smooth_factor;
    let new_smooth_y = smooth_y + dy * active_smooth_factor;

    if new_smooth_x as i32 != smooth_x as i32 || new_smooth_y as i32 != smooth_y as i32 {
        let _ = enigo.move_mouse(new_smooth_x as i32, new_smooth_y as i32, enigo::Coordinate::Abs);
    }

    (new_smooth_x, new_smooth_y, current_face_weight)
}

fn get_left_eye_height(landmarks: &[Landmark]) -> f32 {
    landmarks[LEFT_EYE_BOTTOM].y - landmarks[LEFT_EYE_TOP].y
}

fn check_blink(left_eye_height: f32, threshold: f32) -> bool {
    left_eye_height < threshold
}

fn display_text(frame: &mut core::Mat, text1: &str, text2: &str) {
    let _ = imgproc::put_text(frame, text1, Point::new(30, 40), imgproc::FONT_HERSHEY_SIMPLEX, 0.8, Scalar::new(0.0, 255.0, 255.0, 0.0), 2, imgproc::LINE_8, false);
    if !text2.is_empty() {
        let _ = imgproc::put_text(frame, text2, Point::new(30, 75), imgproc::FONT_HERSHEY_SIMPLEX, 0.8, Scalar::new(0.0, 255.0, 255.0, 0.0), 2, imgproc::LINE_8, false);
    }
}

fn main() -> opencv::Result<()> {
    download_model_if_needed();

    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_DSHOW).unwrap();
    cam.set(videoio::CAP_PROP_FRAME_WIDTH, 1280.0)?;
    cam.set(videoio::CAP_PROP_FRAME_HEIGHT, 720.0)?;

    let mut enigo = Enigo::new(&Settings::default()).unwrap();
    let (screen_w_int, screen_h_int) = enigo.main_display().unwrap();
    let screen_w = screen_w_int as f32;
    let screen_h = screen_h_int as f32;

    let mut smooth_x = screen_w / 2.0;
    let mut smooth_y = screen_h / 2.0;
    let mut smooth_factor = 0.15;
    let mut calibration_stage = 0;
    let mut show_visuals = true;
    let mut invert_face_x = false;
    let mut invert_face_y = false;

    let mut calib_eyes = Calibration { c: (0.5, 0.5), tl: (0.35, 0.3), tr: (0.65, 0.3), bl: (0.35, 0.7), br: (0.65, 0.7) };
    let mut calib_face = Calibration { c: (0.5, 0.5), tl: (0.45, 0.45), tr: (0.55, 0.45), bl: (0.45, 0.55), br: (0.55, 0.55) };

    let mut blink_threshold = 0.004;
    let mut current_face_weight_display = 0.0;

    let mut frame = core::Mat::default();

    loop {
        cam.read(&mut frame)?;
        if frame.empty() { break; }

        let mut flipped_frame = core::Mat::default();
        core::flip(&frame, &mut flipped_frame, 1)?;
        let frame_w = flipped_frame.cols() as f32;
        let frame_h = flipped_frame.rows() as f32;


        let mut landmarks = vec![Landmark { x: 0.5, y: 0.5 }; 478];


        landmarks[RIGHT_EYE_LEFT].x = 0.4;
        landmarks[RIGHT_EYE_RIGHT].x = 0.6;
        landmarks[LEFT_EYE_LEFT].x = 0.4;
        landmarks[LEFT_EYE_RIGHT].x = 0.6;

        let has_face = true;

        if has_face {
            if let Some((avg_x, avg_y, _irises, _bounds)) = calculate_average_gaze(&landmarks) {
                let (yaw, pitch) = get_head_pose(&landmarks);
                let current_eye_height = get_left_eye_height(&landmarks);

                let stages = [
                    "1. Look DEAD CENTER of screen", "2. Look TOP-LEFT (Move EYES only)",
                    "3. Look TOP-RIGHT (Move EYES only)", "4. Look BOTTOM-LEFT (Move EYES only)",
                    "5. Look BOTTOM-RIGHT (Move EYES only)", "6. Turn HEAD slightly TOP-LEFT",
                    "7. Turn HEAD slightly TOP-RIGHT", "8. Turn HEAD slightly BOTTOM-LEFT",
                    "9. Turn HEAD slightly BOTTOM-RIGHT", "10. Look naturally (Eyes open)",
                    "11. Close LEFT EYE completely"
                ];

                if calibration_stage < 11 {
                    display_text(&mut flipped_frame, stages[calibration_stage], "Press 'c' to set");
                    if calibration_stage == 0 {
                        let center_px_x = (frame_w / 2.0) as i32;
                        let center_px_y = (frame_h / 2.0) as i32;
                        let _ = imgproc::circle(&mut flipped_frame, Point::new(center_px_x, center_px_y), 8, Scalar::new(0.0, 0.0, 255.0, 0.0), -1, imgproc::LINE_8, 0);
                    }
                } else {
                    if show_visuals {
                        display_text(&mut flipped_frame, &format!("Dyn Weight: {:.2} | InvY: {}", current_face_weight_display, invert_face_y), "Press 'x'/'y' to invert, 'v' to hide UI");
                    }
                }

                if check_blink(current_eye_height, blink_threshold) {
                    let _ = enigo.button(enigo::Button::Left, enigo::Direction::Click);
                    std::thread::sleep(std::time::Duration::from_millis(500));
                    continue;
                }

                if calibration_stage >= 11 {
                    let result = update_mouse_position(
                        avg_x, avg_y, yaw, pitch, screen_w, screen_h, smooth_x, smooth_y,
                        &calib_eyes, &calib_face, smooth_factor, invert_face_x, invert_face_y, &mut enigo
                    );
                    smooth_x = result.0;
                    smooth_y = result.1;
                    current_face_weight_display = result.2;
                }
            }
        }

        highgui::imshow("Eye Controlled Mouse", &flipped_frame)?;

        let key = highgui::wait_key(1)?;
        if key > 0 {
            let key_char = (key as u8) as char;
            match key_char {
                'q' => break,
                'r' => { calibration_stage = 0; },
                'v' if calibration_stage == 11 => show_visuals = !show_visuals,
                'x' if calibration_stage == 11 => invert_face_x = !invert_face_x,
                'y' if calibration_stage == 11 => invert_face_y = !invert_face_y,
                '=' | '+' => smooth_factor = (smooth_factor + 0.05).min(1.0),
                '-' => smooth_factor = (smooth_factor - 0.05).max(0.01),
                'c' => {
                    if calibration_stage < 11 { calibration_stage += 1; }
                },
                _ => {}
            }
        }
    }
    Ok(())
}