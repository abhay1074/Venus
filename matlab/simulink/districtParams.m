function p = districtParams(varargin)
%DISTRICTPARAMS  Parameter struct for the district screening simulation.
%   p = districtParams() returns the defaults coupled to the locked operating
%   point (sensitivity / specificity from config/operating_point.json).
%   p = districtParams('ophthalmologists', 5, 'camerasPerPhc', 2) overrides.
%   Same fields as backend/venus/stage4_simulate.Params (camelCase).

    p = struct( ...
        'annualPatients', 100000, 'workingDays', 250, 'referablePrevalence', 0.06, 'anyDrPrevalence', 0.18, ...
        'phcs', 30, 'camerasPerPhc', 1, 'operatorsPerPhc', 1, 'captureMinutes', 4.0, 'retakeProbability', 0.05, ...
        'phcHoursPerDay', 8.0, 'aiEnabled', true, 'sensitivity', 0.90, 'specificity', 0.60, 'flagRate', 0.12, ...
        'humanReaderSensitivity', 0.85, 'humanReaderSpecificity', 0.90, 'ophthalmologists', 3, ...
        'doctorHoursPerDay', 6.0, 'teleReviewMinutes', 3.0, 'inPersonMinutes', 20.0, ...
        'operatorHourCost', 150, 'doctorHourCost', 1500, 'cameraAnnualCost', 50000, 'seed', 42, 'sampleFraction', 1.0);
    try
        point = drscreen.operatingPoint();
        at = point.externalTest.at_locked_threshold;
        p.sensitivity = at.sensitivity; p.specificity = at.specificity;
    catch
        % No operating point on this machine: keep the defaults, say so.
        warning('districtParams:noOperatingPoint', 'operating_point.json not found; using default sensitivity/specificity');
    end
    % Flag and retake rates measured on validation images (backend.eval.flag_rate);
    % the served review rule is the validation-chosen policy, whose flag rate on
    % the same sample is what the programme absorbs. Mirrors
    % stage4_simulate.measured_flag_rates.
    configDir = fullfile(drscreen.repoRoot(), 'backend', 'config');
    flagsPath = fullfile(configDir, 'validation_flags.json');
    if isfile(flagsPath)
        v = jsondecode(fileread(flagsPath));
        p.flagRate = v.flag_rate; p.retakeProbability = v.retake_rate;
        policyPath = fullfile(configDir, 'review_policy.json');
        if isfile(policyPath)
            pol = jsondecode(fileread(policyPath));
            if isfield(pol, 'resulting_flag_rate'), p.flagRate = pol.resulting_flag_rate; end
        end
    end
    for i = 1:2:numel(varargin)
        p.(varargin{i}) = varargin{i + 1};
    end
end
