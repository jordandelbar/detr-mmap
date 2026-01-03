use bridge::{SentryControl, SentryMode};
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Controller starting...");

    // Open shared memory sentry control
    let sentry = SentryControl::new("/dev/shm/bridge_sentry_control")?;
    println!("Connected to sentry control\n");

    // Get initial status
    let initial_mode = sentry.get_mode();
    println!("Initial mode: {:?}\n", initial_mode);

    // Toggle between modes every 8 seconds
    let mut is_alarmed = false;
    loop {
        // Toggle mode
        let new_mode = if is_alarmed {
            SentryMode::Standby
        } else {
            SentryMode::Alarmed
        };
        is_alarmed = !is_alarmed;

        println!("Switching to {:?} mode...", new_mode);
        sentry.set_mode(new_mode);

        let current = sentry.get_mode();
        println!("Current mode: {:?}\n", current);

        // Wait 8 seconds before next toggle
        println!("Waiting 8 seconds before next mode change...\n");
        std::thread::sleep(Duration::from_secs(8));
    }
}
