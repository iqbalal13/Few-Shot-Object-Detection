# GOOGLE DRIVE PREFLIGHT — authenticate before expensive training.
if BACKUP_TO_DRIVE:
    from google.colab import drive
    from pathlib import Path
    import uuid

    drive.mount('/content/drive')
    assert Path('/content/drive/MyDrive').is_dir(), 'Google Drive mount tidak tersedia.'
    DRIVE_STAGE1.mkdir(parents=True, exist_ok=True)

    # Unique temporary probe: do not alter any existing user file.
    probe = DRIVE_STAGE1 / ('.stage1_backup_probe_' + uuid.uuid4().hex + '.tmp')
    try:
        probe.write_text('Drive backup preflight OK', encoding='utf-8')
        assert probe.read_text(encoding='utf-8') == 'Drive backup preflight OK'
    finally:
        if probe.is_file():
            probe.unlink()
    print('DRIVE AUTHENTICATION + WRITE TEST PASS:', DRIVE_STAGE1)
else:
    print('Automatic Google Drive backup disabled by configuration.')
