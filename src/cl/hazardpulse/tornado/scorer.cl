// HazardPulse Tornado Scoring Pipeline
// Written in Coherence Lang — the full pipeline from data to prediction.
//
// UNMAINTAINED research twin: NOT the served path, and missing the 2026-10 audit fixes.
// See ../README.md before trusting any number it produces. As written, it:
// 1. Fetches ProbSevere storm objects from NOAA
// 2. Loads pre-trained GBT model weights
// 3. Computes coherence field from atmospheric data
// 4. Scores each storm with tornado probability
// 5. Generates static HTML output
//
// Compiled to WASM for Cloudflare Workers deployment.
// JavaScript is ONLY used for MapLibre maps and D3 charts.

module hazardpulse.tornado.scorer;

import std.core {Option, Result, Ok, Err, Vec, HashMap, String}
import std.io.net.http.client {HttpClient, get}
import std.io.net.http.request {Request}
import std.data.json {JsonValue, parse_json, to_json_string}
import std.io.fs.file {File, read_to_string, write_string}
import std.math.arithmetic {sqrt, abs, exp, max, min}

import hazardpulse.tornado.gbt {GBTModel, sigmoid}
import hazardpulse.tornado.coherence {
    CoherenceDiagnostics, GridField,
    build_coherence_fields, extract_at_point,
    N_LAT, N_LON, latlon_to_cell
}

/// ProbSevere storm object
struct Storm {
    id: USize,
    lat: F64,
    lon: F64,
    mucape: F64,
    mlcape: F64,
    mlcin: F64,
    ebshear: F64,
    srh01: F64,
    mesh: F64,
    vil_density: F64,
    flash_rate: F64,
    flash_density: F64,
    maxllaz: F64,
    p98llaz: F64,
    p98mlaz: F64,
    lja: F64,
    ps: F64,
    size: F64,
    motion_east: F64,
    motion_south: F64
}

/// Scored storm with tornado probability and coherence diagnostics
struct ScoredStorm {
    storm: Storm,
    tornado_probability: F64,
    risk_band: String,
    location_name: String,
    coherence: CoherenceDiagnostics,
    scoring_tier: String
}

/// Risk band classification
fn classify_risk(prob: F64) -> String @ L0 {
    if prob > 0.50 { "high".to_string() }
    else if prob > 0.30 { "elevated".to_string() }
    else if prob > 0.15 { "guarded".to_string() }
    else if prob > 0.05 { "low".to_string() }
    else { "none".to_string() }
}

/// Approximate location name from lat/lon
fn location_name(lat: F64, lon: F64) -> String @ L0 {
    // Major US cities for proximity labeling
    let cities = vec![
        (35.22, -97.44, "Oklahoma City, OK"),
        (32.78, -96.80, "Dallas, TX"),
        (39.10, -94.58, "Kansas City, MO"),
        (41.88, -87.63, "Chicago, IL"),
        (33.75, -84.39, "Atlanta, GA"),
        (36.16, -86.78, "Nashville, TN"),
        (39.77, -86.16, "Indianapolis, IN"),
        (39.96, -83.00, "Columbus, OH"),
        (42.33, -83.05, "Detroit, MI"),
        (44.98, -93.27, "Minneapolis, MN"),
        (38.63, -90.20, "St. Louis, MO"),
    ];

    let mut closest_name = "";
    let mut closest_dist = 999.0;

    for (clat, clon, cname) in &cities {
        let d = sqrt((lat - clat) * (lat - clat) + (lon - clon) * (lon - clon)) * 111.0;
        if d < closest_dist {
            closest_dist = d;
            closest_name = cname;
        }
    }

    if closest_dist < 50.0 {
        format!("Near {}", closest_name)
    } else if closest_dist < 150.0 {
        format!("{:.0} mi from {}", closest_dist * 0.621, closest_name)
    } else {
        format!("{:.1}°N, {:.1}°W", lat, abs(lon))
    }
}

/// Parse ProbSevere GeoJSON into Storm objects
fn parse_probsevere(json: &JsonValue) -> Vec[Storm] @ L0 {
    let mut storms = Vec::new();

    let features = match json.get("features").and_then(|v| v.as_array()) {
        Some(f) => f,
        None => return storms,
    };

    for feature in features {
        let props = match feature.get("properties").and_then(|v| v.as_object()) {
            Some(p) => p,
            None => continue,
        };

        let coords = feature.get("geometry")
            .and_then(|g| g.get("coordinates"))
            .and_then(|c| c.as_array());

        let (lat, lon) = match coords {
            Some(ring) if !ring.is_empty() => {
                let first_ring = ring[0].as_array().unwrap_or(&ring);
                let mut sum_lat = 0.0;
                let mut sum_lon = 0.0;
                let mut count = 0;
                for pt in first_ring {
                    if let Some(coord) = pt.as_array() {
                        if coord.len() >= 2 {
                            sum_lon += coord[0].as_f64().unwrap_or(0.0);
                            sum_lat += coord[1].as_f64().unwrap_or(0.0);
                            count += 1;
                        }
                    }
                }
                if count > 0 { (sum_lat / count as F64, sum_lon / count as F64) }
                else { continue }
            },
            _ => continue,
        };

        fn get_f64(props: &HashMap[String, JsonValue], key: &str) -> F64 {
            props.get(key)
                .and_then(|v| v.as_f64().or_else(|| v.as_str().and_then(|s| s.parse::<F64>().ok())))
                .unwrap_or(0.0)
        }

        storms.push(Storm {
            id: get_f64(props, "ID") as USize,
            lat, lon,
            mucape: get_f64(props, "MUCAPE"),
            mlcape: get_f64(props, "MLCAPE"),
            mlcin: get_f64(props, "MLCIN"),
            ebshear: get_f64(props, "EBSHEAR"),
            srh01: get_f64(props, "SRH01KM"),
            mesh: get_f64(props, "MESH"),
            vil_density: get_f64(props, "VIL_DENSITY"),
            flash_rate: get_f64(props, "FLASH_RATE"),
            flash_density: get_f64(props, "FLASH_DENSITY"),
            maxllaz: get_f64(props, "MAXLLAZ"),
            p98llaz: get_f64(props, "P98LLAZ"),
            p98mlaz: get_f64(props, "P98MLAZ"),
            lja: get_f64(props, "LJA"),
            ps: get_f64(props, "PS"),
            size: get_f64(props, "SIZE"),
            motion_east: get_f64(props, "MOTION_EAST"),
            motion_south: get_f64(props, "MOTION_SOUTH"),
        });
    }

    storms
}

/// Build feature vector for a storm (matches the Python definitive model)
fn build_features(storm: &Storm, coherence: &CoherenceDiagnostics) -> Vec[F64] @ L0 {
    let mut features = Vec::with_capacity(41);

    // Block P: ProbSevere raw (13 features)
    features.push(storm.mucape);
    features.push(storm.mlcape);
    features.push(storm.mlcin);
    features.push(storm.ebshear);
    features.push(storm.srh01);
    features.push(storm.mesh);
    features.push(storm.vil_density);
    features.push(storm.flash_rate);
    features.push(storm.maxllaz);
    features.push(storm.p98llaz);
    features.push(storm.p98mlaz);
    features.push(storm.ps);
    features.push(storm.size);

    // Block E: Storm evolution (6 features) — placeholder for now
    features.push(0.0); // storm_age_min
    features.push(1.0); // maxllaz_trend
    features.push(1.0); // flash_rate_trend
    features.push(1.0); // ps_trend
    features.push(0.0); // sustained_rotation_min
    let speed = sqrt(storm.motion_east * storm.motion_east
                   + storm.motion_south * storm.motion_south);
    features.push(speed); // storm_speed_ms

    // Block H: HRRR atmospheric context (12 features)
    // Use ProbSevere values as proxy when HRRR unavailable
    features.push(storm.mucape);   // hrrr_cape
    features.push(storm.mlcin);    // hrrr_cin
    features.push(storm.srh01);    // hrrr_srh01
    features.push(storm.srh01 * 1.5); // hrrr_srh03 (estimate)
    let shear_06 = storm.ebshear * 0.5144; // kt to m/s
    features.push(shear_06);       // hrrr_shear06
    let shear_01 = shear_06 * 0.4; // estimate
    features.push(shear_01);       // hrrr_shear01
    features.push(0.0);            // hrrr_pwat (not in ProbSevere)
    features.push(288.0);          // hrrr_t2m (default)
    features.push(280.0);          // hrrr_td2m (default)
    features.push(35.0);           // hrrr_refc (default)
    // STP estimate
    // One expression in parentheses: a continuation line that starts with '*' is a NEW
    // statement in Coherence, so until 2026-10-02 the SRH and shear factors were dropped.
    let stp = ((storm.mucape / 1500.0).min(2.0)
            * (storm.srh01.abs() / 150.0).min(2.0)
            * (storm.ebshear / 20.0).min(2.0));
    features.push(stp);            // hrrr_stp
    let srh05 = storm.srh01 * 0.5; // estimate 0-500m SRH
    features.push(srh05);          // hrrr_srh05_est

    // Block C: Coherence Field Theory (10 features)
    features.push(coherence.tau);
    features.push(coherence.grad_tau);
    features.push(coherence.torsion);
    features.push(coherence.alignment);
    features.push(coherence.s_over_gamma);
    features.push(coherence.da);
    features.push(coherence.tau * storm.maxllaz * 100.0); // tau_x_maxllaz
    features.push(coherence.alignment * storm.mucape / 1000.0); // alignment_x_cape
    features.push(coherence.torsion * storm.srh01 / 100.0); // torsion_x_srh
    features.push(coherence.singularity_count as F64); // singularity_count

    features
}

/// Score all storms and return sorted results
fn score_storms(
    storms: Vec[Storm],
    model: &GBTModel,
    tau_field: &GridField,
    grad_tau_field: &GridField,
    source_field: &GridField,
    gamma_field: &GridField
) -> Vec[ScoredStorm] @ L0 {
    let mut scored = Vec::with_capacity(storms.len());

    for storm in storms {
        // Extract coherence at storm location
        let coherence = extract_at_point(
            storm.lat, storm.lon,
            tau_field, grad_tau_field,
            source_field, gamma_field,
            storm.srh01, storm.ebshear
        );

        // Build feature vector
        let features = build_features(&storm, &coherence);

        // Predict with GBT
        let prob = model.predict_proba(&features);

        // Classify risk
        let risk = classify_risk(prob);

        // Get location name
        let loc = location_name(storm.lat, storm.lon);

        scored.push(ScoredStorm {
            storm,
            tornado_probability: prob,
            risk_band: risk,
            location_name: loc,
            coherence,
            scoring_tier: "tier1_ml".to_string(),
        });
    }

    // Sort by probability descending
    scored.sort_by(|a, b| b.tornado_probability.partial_cmp(&a.tornado_probability).unwrap());

    scored
}

/// Main entry point — fetch, score, output
pub fn run() -> Result[(), String] @ L1 {
    // 1. Load model
    let model_json = read_to_string("results/models/tornado_gbt_v1.json")
        .map_err(|e| format!("Failed to load model: {}", e))?;
    let model = GBTModel::from_json_str(&model_json)?;

    // 2. Fetch ProbSevere
    let ps_response = get("https://noaa-mrms-pds.s3.amazonaws.com/ProbSevere/latest.json")
        .map_err(|e| format!("ProbSevere fetch failed: {}", e))?;
    let ps_json = parse_json(&ps_response.body)
        .map_err(|e| format!("ProbSevere parse failed: {}", e))?;
    let storms = parse_probsevere(&ps_json);

    // 3. Build coherence fields from storm atmospheric data
    let mut cape_field = GridField::new(N_LAT, N_LON);
    let mut srh_field = GridField::new(N_LAT, N_LON);
    let mut shear_field = GridField::new(N_LAT, N_LON);
    let mut cin_field = GridField::new(N_LAT, N_LON);
    let mut td_field = GridField::new(N_LAT, N_LON);

    for storm in &storms {
        let (i, j) = latlon_to_cell(storm.lat, storm.lon);
        cape_field.set(i, j, max(cape_field.get(i, j), storm.mucape));
        srh_field.set(i, j, max(srh_field.get(i, j), storm.srh01.abs()));
        shear_field.set(i, j, max(shear_field.get(i, j), storm.ebshear));
        cin_field.set(i, j, max(cin_field.get(i, j), storm.mlcin.abs()));
        td_field.set(i, j, 5.0); // default Td depression
    }

    let (tau, grad_tau, source, gamma) = build_coherence_fields(
        &cape_field, &srh_field, &shear_field, &cin_field, &td_field
    );

    // 4. Score all storms
    let scored = score_storms(storms, &model, &tau, &grad_tau, &source, &gamma);

    // 5. Output JSON
    let output = scored_to_json(&scored);
    write_string("dist/data/live-tornadoes.json", &output)
        .map_err(|e| format!("Write failed: {}", e))?;

    Ok(())
}

/// Convert scored storms to JSON string
fn scored_to_json(scored: &Vec[ScoredStorm]) -> String @ L0 {
    let mut storms_json = Vec::new();
    for s in scored {
        storms_json.push(format!(
            r#"{{"storm_id":{},"lat":{:.4},"lon":{:.4},"tornado_probability":{:.4},"risk_band":"{}","location":"{}","mucape":{:.0},"srh01":{:.0},"maxllaz":{:.4},"tau":{:.4},"singularity":{}}}"#,
            s.storm.id, s.storm.lat, s.storm.lon,
            s.tornado_probability, s.risk_band, s.location_name,
            s.storm.mucape, s.storm.srh01, s.storm.maxllaz,
            s.coherence.tau, s.coherence.singularity_count
        ));
    }

    format!(
        r#"{{"updated_at":"{}","n_active_storms":{},"model_version":"hp-tornado-coherence-v1","disclaimer":"RESEARCH ONLY. NOT operational.","storms":[{}]}}"#,
        "now", // TODO: use std.time for actual timestamp
        scored.len(),
        storms_json.join(",")
    )
}
