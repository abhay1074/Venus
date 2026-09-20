function point = operatingPoint(configDir)
%OPERATINGPOINT  Read config/operating_point.json and verify its fingerprint.
%   point = drscreen.operatingPoint() reads the locked operating point written
%   by backend/eval/calibrate.py and errors if the calibration manifest on
%   disk does not hash to the fingerprint recorded in it. Serving must not run
%   with a threshold nobody can vouch for.
%
%   Fields: modelVersion, graderTag, calibrationFingerprint, calibration.a,
%   calibration.b, thresholds.referable, thresholds.referable85,
%   thresholds.abstainBand, externalTest (as decoded), raw (the whole file).

    if nargin < 1
        configDir = fullfile(drscreen.repoRoot(), 'backend', 'config');
    end
    raw = jsondecode(fileread(fullfile(configDir, 'operating_point.json')));
    manifest = fullfile(drscreen.repoRoot(), 'backend', 'data', 'manifests', raw.calibration_manifest);
    if ~isfile(manifest)
        error('drscreen:operatingPoint', 'calibration manifest missing: %s', manifest);
    end
    actual = drscreen.sha256File(manifest);
    if ~strcmpi(actual, raw.calibration_fingerprint)
        error('drscreen:operatingPoint', ...
            'calibration manifest fingerprint does not match operating_point.json (%s... vs %s...)', ...
            actual(1:12), raw.calibration_fingerprint(1:12));
    end
    point.modelVersion = raw.model_version;
    point.graderTag = raw.grader_tag;
    point.calibrationFingerprint = raw.calibration_fingerprint;
    point.calibration = struct('a', raw.calibration.a, 'b', raw.calibration.b);
    point.thresholds = struct('referable', raw.thresholds.referable, ...
        'referable85', raw.thresholds.referable_85pc_alternative, ...
        'abstainBand', raw.thresholds.abstain_band);
    point.externalTest = raw.external_test;
    point.raw = raw;
end
