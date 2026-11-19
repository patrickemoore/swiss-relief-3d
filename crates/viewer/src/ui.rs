//! The window's interface.
//!
//! What a visitor needs sits in a bar at the bottom: the places to fly to
//! and the comparison. Everything else is the lab, a panel that stays out of
//! the way until asked for: every parameter as a toggle or slider,
//! conditions and presets, debug views, camera paths and timings.

use egui::{CollapsingHeader, ComboBox, Context, Grid, RichText, ScrollArea, SidePanel, Slider, Ui};

use crate::session::Session;
use relief_render::camera::CameraMode;
use relief_render::labels::{Kind as LabelKind, View as LabelView};
use relief_render::params::{Condition, DebugView, LightMode, ShadingMode, ShadingParams};
use relief_render::path::WHOLE_COUNTRY;
use relief_render::renderer::timers::Pass;
use relief_render::renderer::FrameStats;

const PANEL_WIDTH: f32 = 340.0;
const SIDE_NAMES: [&str; 2] = ["A", "B"];

/// Timings and counts shown in the performance section.
pub struct Performance {
    pub adapter: String,
    pub resolution: [u32; 2],
    /// Mean and worst frame time over the last few seconds, in milliseconds.
    pub mean_frame_ms: f32,
    pub max_frame_ms: f32,
    /// Smoothed GPU time per pass, or `None` without timestamp queries.
    pub pass_ms: Option<[f32; Pass::ALL.len()]>,
    pub stats: FrameStats,
}

/// Gives the interface its look: light, rounded surfaces with room to
/// breathe, in place of the toolkit's dense dark default.
pub fn style(ctx: &Context) {
    let mut visuals = egui::Visuals::light();
    let rounded = egui::CornerRadius::same(8);
    visuals.window_corner_radius = egui::CornerRadius::same(12);
    visuals.menu_corner_radius = rounded;
    for widget in [
        &mut visuals.widgets.noninteractive,
        &mut visuals.widgets.inactive,
        &mut visuals.widgets.hovered,
        &mut visuals.widgets.active,
        &mut visuals.widgets.open,
    ] {
        widget.corner_radius = rounded;
    }
    visuals.selection.bg_fill = egui::Color32::from_rgb(28, 38, 51);
    visuals.selection.stroke.color = egui::Color32::from_rgb(243, 246, 249);
    visuals.panel_fill = egui::Color32::from_rgb(247, 249, 251);
    ctx.set_visuals(visuals);
    ctx.style_mut(|style| {
        style.spacing.item_spacing = egui::vec2(8.0, 6.0);
        style.spacing.button_padding = egui::vec2(9.0, 5.0);
        style.spacing.interact_size.y = 26.0;
    });
}

pub fn draw(ctx: &Context, session: &mut Session, performance: &Performance) {
    draw_labels(ctx, session);
    draw_side_labels(ctx, session);
    if session.show_panel {
        lab(ctx, session, performance);
    }
    // After the panel, so that the bar knows what is left of the window.
    place_bar(ctx, session);
}

/// The bar at the bottom of the view: places to fly to, the comparison and
/// the way into the lab.
fn place_bar(ctx: &Context, session: &mut Session) {
    let free = ctx.available_rect();
    egui::Area::new(egui::Id::new("places"))
        .pivot(egui::Align2::CENTER_BOTTOM)
        .fixed_pos(egui::pos2(free.center().x, free.bottom() - 16.0))
        .show(ctx, |ui| {
            egui::Frame::popup(ui.style()).inner_margin(egui::Margin::same(6)).show(ui, |ui| {
                ui.set_max_width(free.width() - 48.0);
                ui.horizontal_wrapped(|ui| {
                    if ui.selectable_label(session.at_home, "Whole country").clicked() {
                        session.fly_home();
                    }
                    for index in 0..session.places.len() {
                        let place = session.places[index];
                        if place.id == WHOLE_COUNTRY {
                            continue;
                        }
                        if ui.selectable_label(session.place == Some(place.id), place.name).clicked() {
                            session.fly_to_place(index);
                        }
                    }
                    ui.separator();
                    ui.toggle_value(&mut session.split_enabled, "Compare").on_hover_text("Two looks side by side (C)");
                    ui.toggle_value(&mut session.show_labels, "Names").on_hover_text("Peaks, lakes and towns (L)");
                    ui.toggle_value(&mut session.show_trails, "Trails").on_hover_text("Hiking trails (T)");
                    ui.toggle_value(&mut session.show_panel, "Lab").on_hover_text("Every parameter of the method (F1)");
                });
            });
        });
}

/// The panel with every parameter.
fn lab(ctx: &Context, session: &mut Session, performance: &Performance) {
    SidePanel::right("controls").default_width(PANEL_WIDTH).show(ctx, |ui| {
        ui.add_space(4.0);
        ui.heading("The lab");
        ui.label(RichText::new("F1 hides this panel · F12 saves a frame").weak());
        ui.separator();
        ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            compare_section(ui, session);
            conditions_section(ui, session);
            let params = session.edited_params();
            light_section(ui, params);
            local_light_section(ui, params);
            generalization_section(ui, params);
            aerial_section(ui, params);
            color_section(ui, params);
            depth_section(ui, params);
            baseline_section(ui, params);
            debug_section(ui, params);
            camera_section(ui, session);
            presets_section(ui, session);
            performance_section(ui, session, performance);
            help_section(ui);
            if !session.status.is_empty() {
                ui.separator();
                ui.label(RichText::new(&session.status).weak());
            }
        });
    });
}

/// Names the two sides at the top of the 3D view while comparing.
fn draw_side_labels(ctx: &Context, session: &Session) {
    if !session.split_enabled {
        return;
    }
    let width = ctx.screen_rect().width();
    let positions = [12.0, session.split_position * width + 12.0];
    for (side, x) in positions.into_iter().enumerate() {
        egui::Area::new(egui::Id::new(("side label", side))).fixed_pos(egui::pos2(x, 10.0)).interactable(false).show(
            ctx,
            |ui| {
                egui::Frame::popup(ui.style()).show(ui, |ui| {
                    ui.label(
                        RichText::new(format!("{}  {}", SIDE_NAMES[side], session.sides[side].params.mode.label()))
                            .strong(),
                    );
                });
            },
        );
    }
}

fn section(ui: &mut Ui, title: &str, open: bool, contents: impl FnOnce(&mut Ui)) {
    CollapsingHeader::new(RichText::new(title).strong()).default_open(open).show(ui, contents);
}

fn slider(ui: &mut Ui, label: &str, value: &mut f32, range: std::ops::RangeInclusive<f32>, hint: &str) {
    ui.add(Slider::new(value, range).text(label)).on_hover_text(hint);
}

fn compare_section(ui: &mut Ui, session: &mut Session) {
    section(ui, "Compare", true, |ui| {
        ui.checkbox(&mut session.split_enabled, "Split screen (C)")
            .on_hover_text("Show two parameter sets side by side: A on the left, B on the right.");
        if session.split_enabled {
            slider(ui, "divider", &mut session.split_position, 0.05..=0.95, "Where side B begins.");
            ui.horizontal(|ui| {
                ui.label("Panel edits");
                for (side, name) in SIDE_NAMES.into_iter().enumerate() {
                    ui.selectable_value(&mut session.edited_side, side, name);
                }
                let (from, to) = (session.edited_side, 1 - session.edited_side);
                if ui.button(format!("Copy {} to {}", SIDE_NAMES[from], SIDE_NAMES[to])).clicked() {
                    session.sides[to].params = session.sides[from].params.clone();
                }
            });
        } else {
            session.edited_side = 0;
        }
    });
}

fn conditions_section(ui: &mut Ui, session: &mut Session) {
    section(ui, "Looks", true, |ui| {
        if ui
            .button("Showcase")
            .on_hover_text("The method tuned for looks: a lower light, more contrast, less haze.")
            .clicked()
        {
            *session.edited_params() = ShadingParams::showcase();
            session.status = "Showcase look".to_owned();
        }
        ui.label(RichText::new("The conditions of the evaluation:").weak());
        ui.horizontal_wrapped(|ui| {
            for condition in Condition::ALL {
                if ui.button(condition.label()).on_hover_text(condition.description()).clicked() {
                    session.apply_condition(condition);
                }
            }
        });
        ui.label(RichText::new("Keys 1 to 5. Untick a component below to ablate it.").weak());
    });
}

fn light_section(ui: &mut Ui, params: &mut ShadingParams) {
    section(ui, "Base light (6.1)", true, |ui| {
        ComboBox::from_label("shading").selected_text(params.mode.label()).show_ui(ui, |ui| {
            for mode in ShadingMode::ALL {
                ui.selectable_value(&mut params.mode, mode, mode.label());
            }
        });
        ui.horizontal(|ui| {
            ui.radio_value(&mut params.light_mode, LightMode::CameraRelative, "camera-relative");
            ui.radio_value(&mut params.light_mode, LightMode::WorldFixed, "world-fixed");
        });
        slider(
            ui,
            "screen angle",
            &mut params.light_screen_angle_deg,
            0.0..=90.0,
            "How far left of screen-up the light sits. 45 is north-west on a map, 22.5 north-north-west.",
        );
        slider(
            ui,
            "elevation",
            &mut params.light_elevation_deg,
            20.0..=70.0,
            "Angle of the light above the image plane.",
        );
        slider(
            ui,
            "lag (s)",
            &mut params.light_lag_s,
            0.0..=5.0,
            "Time constant with which the light trails the camera. 0 is a rigid headlight.",
        );
    });
}

fn local_light_section(ui: &mut Ui, params: &mut ShadingParams) {
    section(ui, "Local light adjustment (6.2)", false, |ui| {
        ui.checkbox(&mut params.local_light, "Rotate the light per landform");
        ui.add_enabled_ui(params.local_light, |ui| {
            slider(
                ui,
                "max rotation",
                &mut params.local_max_rotation_deg,
                0.0..=60.0,
                "Largest rotation of the light, in degrees.",
            );
            slider(
                ui,
                "acceptable angle",
                &mut params.local_ok_angle_deg,
                20.0..=80.0,
                "Angle between light and fall line that needs no correction.",
            );
            slider(
                ui,
                "blur (m)",
                &mut params.local_blur_m,
                100.0..=2000.0,
                "Standard deviation of the blur, about one landform wide.",
            );
            slider(
                ui,
                "hysteresis",
                &mut params.local_hysteresis_deg,
                0.0..=30.0,
                "How far past 90 degrees a cell keeps rotating the same way.",
            );
            slider(
                ui,
                "sign softness",
                &mut params.local_sign_softness_deg,
                0.5..=30.0,
                "Softness of the switch between rotating one way and the other.",
            );
        });
    });
}

fn generalization_section(ui: &mut Ui, params: &mut ShadingParams) {
    section(ui, "Multi-scale shading (6.3)", false, |ui| {
        ui.checkbox(&mut params.generalization, "View-dependent generalization")
            .on_hover_text("Off shades the finest level everywhere.");
        slider(
            ui,
            "bias",
            &mut params.generalization_bias,
            0.0..=4.0,
            "Generalization bias b: how many levels coarser than a pixel the finest detail is.",
        );
        slider(
            ui,
            "landform level",
            &mut params.landform_level,
            4.0..=9.0,
            "Pyramid level of the landform term. Level 7 is 1.28 km.",
        );
        slider(ui, "landform weight", &mut params.landform_weight, 0.0..=1.0, "Weight kL of the landform term.");
        slider(ui, "detail weight", &mut params.detail_weight, 0.0..=0.6, "Weight k0 of the finest detail term.");
        slider(
            ui,
            "detail falloff",
            &mut params.detail_falloff,
            0.2..=1.0,
            "Each coarser detail term is weighted by this factor again.",
        );
        slider(ui, "detail gain", &mut params.detail_gain, 1.0..=10.0, "Exaggeration a of the detail terms.");
        slider(
            ui,
            "shading exaggeration",
            &mut params.shading_exaggeration,
            1.0..=4.0,
            "Vertical exaggeration of the gradients used for shading.",
        );
        slider(
            ui,
            "geometry exaggeration",
            &mut params.geometry_exaggeration,
            1.0..=2.0,
            "Vertical exaggeration of the mesh.",
        );
        slider(ui, "flat brightness", &mut params.flat_brightness, 0.5..=0.95, "Brightness of flat ground.");
        slider(ui, "contrast", &mut params.contrast, 0.3..=2.0, "Tone mapping contrast around flat ground.");
        ui.checkbox(&mut params.analytic_footprint, "Analytic pixel footprint")
            .on_hover_text("Avoids blurry seams along silhouettes, where screen-space derivatives jump.");
        ui.checkbox(&mut params.accentuation, "Ridge accentuation (4.3)")
            .on_hover_text("Use the pyramids that keep major ridges and valleys sharp.");
    });
}

fn aerial_section(ui: &mut Ui, params: &mut ShadingParams) {
    section(ui, "Aerial perspective (7.1)", false, |ui| {
        ui.checkbox(&mut params.elevation_contrast, "Elevation contrast");
        ui.add_enabled_ui(params.elevation_contrast, |ui| {
            slider(
                ui,
                "valley tone",
                &mut params.valley_tone,
                0.3..=0.9,
                "Medium tone B_mid the lowlands are compressed toward.",
            );
            slider(
                ui,
                "low contrast",
                &mut params.low_contrast,
                0.0..=1.0,
                "Contrast C_low kept in the lowlands. Raise toward 0.7 when height fog is on.",
            );
            slider(
                ui,
                "from elevation",
                &mut params.contrast_elevation_lo,
                0.0..=1.0,
                "Normalized elevation where contrast starts to rise.",
            );
            slider(
                ui,
                "to elevation",
                &mut params.contrast_elevation_hi,
                0.0..=1.0,
                "Normalized elevation of full contrast.",
            );
        });
        ui.checkbox(&mut params.fog, "Height fog");
        ui.add_enabled_ui(params.fog, |ui| {
            slider(
                ui,
                "density (1/km)",
                &mut params.fog_density_per_km,
                0.0..=0.4,
                "Haze density at the lowland level.",
            );
            slider(
                ui,
                "scale height (m)",
                &mut params.fog_scale_height_m,
                200.0..=4000.0,
                "Height over which the density falls by a factor e.",
            );
            slider(ui, "max haze", &mut params.fog_max, 0.0..=1.0, "Cap on the haze, so the horizon keeps some form.");
        });
    });
}

fn color_section(ui: &mut Ui, params: &mut ShadingParams) {
    section(ui, "Colour (7.2)", false, |ui| {
        ui.checkbox(&mut params.color, "Colour table").on_hover_text("Off shows the grey shading.");
        ui.checkbox(&mut params.water, "Lakes");
        color_picker(ui, "haze and horizon", &mut params.haze_color);
        color_picker(ui, "zenith", &mut params.zenith_color);
        slider(
            ui,
            "edge fade (m)",
            &mut params.edge_fade_m,
            0.0..=5000.0,
            "Width over which the terrain dissolves into the sky at the edge of its data.",
        );
    });
}

/// Edits an sRGB colour stored as floats.
fn color_picker(ui: &mut Ui, label: &str, color: &mut [f32; 3]) {
    ui.horizontal(|ui| {
        let mut bytes = color.map(|channel| (channel * 255.0).round() as u8);
        if ui.color_edit_button_srgb(&mut bytes).changed() {
            *color = bytes.map(|channel| channel as f32 / 255.0);
        }
        ui.label(label);
    });
}

fn depth_section(ui: &mut Ui, params: &mut ShadingParams) {
    section(ui, "Depth cues (7.3)", false, |ui| {
        ui.checkbox(&mut params.depth_edges, "Depth-edge halos");
        ui.add_enabled_ui(params.depth_edges, |ui| {
            slider(
                ui,
                "strength",
                &mut params.depth_edge_strength,
                0.0..=0.5,
                "Keep it low: the goal is separation, not outlines.",
            );
            slider(
                ui,
                "radius",
                &mut params.depth_edge_radius,
                0.002..=0.04,
                "Blur radius as a fraction of the image height.",
            );
        });
        ui.checkbox(&mut params.silhouettes, "Silhouette lines")
            .on_hover_text("Thin lines at large depth jumps. Swiss maps use none.");
    });
}

fn baseline_section(ui: &mut Ui, params: &mut ShadingParams) {
    section(ui, "Realistic baseline", false, |ui| {
        slider(ui, "sun azimuth", &mut params.sun_azimuth_deg, 0.0..=360.0, "Compass direction of the sun.");
        slider(ui, "sun elevation", &mut params.sun_elevation_deg, 5.0..=85.0, "Height of the sun above the horizon.");
        slider(ui, "ambient", &mut params.sun_ambient, 0.0..=0.6, "Share of light that reaches shadows.");
        ui.checkbox(&mut params.shadows, "Cast shadows");
    });
}

fn debug_section(ui: &mut Ui, params: &mut ShadingParams) {
    section(ui, "Debug view", false, |ui| {
        ComboBox::from_id_salt("debug view").width(220.0).selected_text(params.debug_view.label()).show_ui(ui, |ui| {
            for view in DebugView::ALL {
                ui.selectable_value(&mut params.debug_view, view, view.label());
            }
        });
    });
}

fn camera_section(ui: &mut Ui, session: &mut Session) {
    section(ui, "Camera and paths", true, |ui| {
        ui.horizontal(|ui| {
            let mut mode = session.controller.mode;
            ui.radio_value(&mut mode, CameraMode::Orbit, "orbit");
            ui.radio_value(&mut mode, CameraMode::Fly, "fly");
            ui.label(RichText::new("(Tab)").weak());
            if mode != session.controller.mode {
                session.set_camera_mode(mode);
            }
            if mode == CameraMode::Fly {
                ui.label(format!("{:.0} m/s", session.controller.fly_speed));
            }
        });
        let mut fov = session.controller.fov_y.to_degrees();
        slider(ui, "field of view", &mut fov, 20.0..=90.0, "Vertical field of view in degrees.");
        session.controller.fov_y = fov.to_radians();
        slider(
            ui,
            "mesh detail (px)",
            &mut session.pixels_per_quad,
            2.0..=16.0,
            "Target size of a mesh quad on screen. Shading does not depend on it.",
        );

        ui.label("Viewpoints");
        ui.horizontal_wrapped(|ui| {
            for index in 0..session.viewpoints.len() {
                if ui.small_button(session.viewpoints[index].name).clicked() {
                    session.jump_to(session.viewpoints[index].pose);
                }
            }
        });

        ui.label("Flights");
        ui.horizontal_wrapped(|ui| {
            for index in 0..session.flights.len() {
                if ui.small_button(&session.flights[index].name).clicked() {
                    session.play(session.flights[index].clone());
                }
            }
            for index in 0..session.saved_paths.len() {
                let file = session.saved_paths[index].clone();
                let name = file.file_stem().map_or_else(String::new, |stem| stem.to_string_lossy().into_owned());
                if ui.small_button(name).on_hover_text(file.display().to_string()).clicked() {
                    session.play_file(&file);
                }
            }
        });
        if let Some(playback) = &session.playback {
            let progress = playback.time / playback.path.duration().max(1e-3);
            ui.add(egui::ProgressBar::new(progress).text(format!("{} (Esc stops)", playback.path.name)));
        }

        ui.horizontal(|ui| {
            let label = match &session.recorder {
                Some(recorder) => format!("Stop recording ({:.0} s)", recorder.elapsed()),
                None => "Record".to_owned(),
            };
            if ui.button(label).clicked() {
                session.toggle_recording();
            }
            if session.recording.is_some() {
                ui.add(egui::TextEdit::singleline(&mut session.path_name).desired_width(90.0));
                if ui.button("Save").clicked() {
                    session.save_recording();
                }
            }
        });
    });
}

fn presets_section(ui: &mut Ui, session: &mut Session) {
    section(ui, "Presets", false, |ui| {
        ui.horizontal(|ui| {
            ui.add(egui::TextEdit::singleline(&mut session.preset_name).desired_width(140.0));
            if ui.button("Save").clicked() {
                session.save_preset();
            }
        });
        ui.horizontal_wrapped(|ui| {
            for index in 0..session.presets.len() {
                let file = session.presets[index].clone();
                let name = file.file_stem().map_or_else(String::new, |stem| stem.to_string_lossy().into_owned());
                if ui.small_button(name).clicked() {
                    session.load_preset(&file);
                }
            }
        });
    });
}

fn performance_section(ui: &mut Ui, session: &mut Session, performance: &Performance) {
    section(ui, "Performance", true, |ui| {
        let [width, height] = performance.resolution;
        ui.label(format!(
            "{:.0} fps mean, {:.0} fps worst at {width} x {height}",
            1000.0 / performance.mean_frame_ms.max(1e-3),
            1000.0 / performance.max_frame_ms.max(1e-3)
        ));
        ui.label(RichText::new(&performance.adapter).weak());
        ui.label(format!(
            "{} patches, {:.2} M triangles",
            performance.stats.patches,
            performance.stats.triangles as f32 / 1e6
        ));
        if performance.stats.missing_tiles > 0 {
            ui.label(format!("streaming {} terrain tiles", performance.stats.missing_tiles));
        }
        match performance.pass_ms {
            Some(pass_ms) => {
                Grid::new("pass timings").num_columns(2).striped(true).show(ui, |ui| {
                    for pass in Pass::ALL {
                        ui.label(pass.label());
                        ui.label(format!("{:.2} ms", pass_ms[pass as usize]));
                        ui.end_row();
                    }
                    ui.label(RichText::new("GPU total").strong());
                    ui.label(RichText::new(format!("{:.2} ms", pass_ms.iter().sum::<f32>())).strong());
                    ui.end_row();
                });
            }
            None => {
                ui.label(RichText::new("This GPU does not report per-pass timings.").weak());
            }
        }
        ui.checkbox(&mut session.vsync, "Limit to display refresh rate");
        if ui.button("Save frame (F12)").clicked() {
            session.capture_requested = true;
        }
    });
}

fn help_section(ui: &mut Ui) {
    section(ui, "Controls", false, |ui| {
        let rows = [
            ("Orbit", "drag pans, right-drag rotates and tilts, wheel zooms"),
            ("", "WASD pans, arrow keys rotate and tilt"),
            ("Fly", "drag looks around, WASD moves, Q and E sink and rise"),
            ("", "wheel changes speed, Shift goes faster"),
            ("Tab", "switch between orbit and fly"),
            ("1 to 5", "conditions A to E"),
            ("C", "split-screen comparison"),
            ("Esc", "stop a flight"),
        ];
        Grid::new("controls help").num_columns(2).show(ui, |ui| {
            for (key, action) in rows {
                ui.label(RichText::new(key).strong());
                ui.label(action);
                ui.end_row();
            }
        });
    });
}

/// Paints the names of peaks, lakes and towns over the view.
fn draw_labels(ctx: &Context, session: &mut Session) {
    if !session.show_labels {
        return;
    }
    let pixels_per_point = ctx.pixels_per_point();
    let size = ctx.screen_rect().size();
    let view = LabelView {
        pose: session.controller.pose(),
        fov_y: session.controller.fov_y,
        viewport: glam::Vec2::new(size.x, size.y) * pixels_per_point,
        pixel_ratio: pixels_per_point,
        z_scale: session.z_scale(),
        ground: &session.ground,
        // The place bar at the bottom.
        insets: [0.0, 0.0, 56.0 * pixels_per_point, 0.0],
    };
    let dt = ctx.input(|input| input.stable_dt).min(0.1);
    session.labels.update(&view, dt);
    if session.labels.animating() {
        ctx.request_repaint();
    }

    let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Background, egui::Id::new("labels")));
    for placed in session.labels.placed() {
        let to_points = |x: f32, y: f32| egui::pos2(x / pixels_per_point, y / pixels_per_point);
        let (ink, halo) = match placed.kind {
            LabelKind::Lake | LabelKind::Glacier => {
                (egui::Color32::from_rgb(28, 78, 140), egui::Color32::from_rgb(236, 243, 250))
            }
            _ => (egui::Color32::from_rgb(28, 32, 38), egui::Color32::from_rgb(248, 247, 242)),
        };
        let ink = ink.gamma_multiply(placed.opacity);
        let halo = halo.gamma_multiply(0.85 * placed.opacity);
        let font = egui::FontId::proportional(placed.font / pixels_per_point);
        let point = to_points(placed.x, placed.y);
        let [left, top, width, _] = placed.rect;
        let anchor = to_points(left + 0.5 * width, top);
        let text = |painter: &egui::Painter, pos: egui::Pos2, align: egui::Align2, text: &str, font: egui::FontId| {
            // A halo: the text in the paper colour around the text itself.
            for offset in
                [(-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0), (-0.7, -0.7), (0.7, -0.7), (-0.7, 0.7), (0.7, 0.7)]
            {
                painter.text(pos + egui::vec2(offset.0, offset.1), align, text, font.clone(), halo);
            }
            painter.text(pos, align, text, font.clone(), ink)
        };
        match placed.kind {
            LabelKind::Peak => {
                let rect = text(&painter, anchor, egui::Align2::CENTER_TOP, &placed.name, font.clone());
                if let Some(elevation) = placed.elevation {
                    let small = egui::FontId::proportional(0.8 * placed.font / pixels_per_point);
                    text(
                        &painter,
                        egui::pos2(anchor.x, rect.bottom() - 1.0),
                        egui::Align2::CENTER_TOP,
                        &elevation.to_string(),
                        small,
                    );
                }
                painter.circle_filled(point, 2.2, halo);
                painter.circle_filled(point, 1.4, ink);
            }
            LabelKind::Town => {
                painter.circle_filled(point, 3.6, halo);
                painter.circle_filled(point, 2.4, ink);
                text(
                    &painter,
                    to_points(left, top + 0.5 * placed.rect[3]),
                    egui::Align2::LEFT_CENTER,
                    &placed.name,
                    font,
                );
            }
            LabelKind::Lake | LabelKind::Glacier => {
                text(
                    &painter,
                    to_points(left + 0.5 * width, top + 0.5 * placed.rect[3]),
                    egui::Align2::CENTER_CENTER,
                    &placed.name,
                    font,
                );
            }
        }
    }
}
