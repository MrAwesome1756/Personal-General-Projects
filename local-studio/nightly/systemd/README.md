# Nightly on Linux (systemd user timer)

```bash
mkdir -p ~/.config/systemd/user
cp nightly/systemd/local-studio-nightly.{service,timer} ~/.config/systemd/user/
# edit WorkingDirectory/ExecStart in the .service to this repo's absolute path
systemctl --user daemon-reload
systemctl --user enable --now local-studio-nightly.timer
loginctl enable-linger "$USER"      # keep user timers running when logged out
systemctl --user start local-studio-nightly.service   # test run now
journalctl --user -u local-studio-nightly -f
```
Or with cron: `30 1 * * * /path/to/local-studio/nightly/run_nightly.sh`
