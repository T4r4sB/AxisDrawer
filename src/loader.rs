use crate::solver::*;
use clap::Parser;

/// Simple program to draw axis system by equations
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Name of file to parse
    pub input: String,

    /// Name of output file
    #[arg(short, long)]
    pub output: Option<String>,

    /// Required frame size
    #[arg(short, long)]
    pub size: Option<usize>,

    /// Required frames count
    #[arg(short, long)]
    pub length: Option<usize>,
}

pub fn parse_command_line() -> Args {
    Args::parse()
}

pub fn load(file: &std::path::Path) -> Result<AxisSystem, String> {
      let content = std::fs::read_to_string(file)
        .map_err(|e| format!("Unable to load file {}: {}", file.to_string_lossy(), e))?;

    let axis_system: AxisSystem = serde_json::from_str(&content)
        .map_err(|e| format!("Unable to parse content of {}: {}", file.to_string_lossy(), e))?;
    Ok(axis_system)
}
