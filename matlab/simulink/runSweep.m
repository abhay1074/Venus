function sweep = runSweep(opts)
%RUNSWEEP  Cameras x ophthalmologists x AI operating point, Pareto front.
%   sweep = runSweep() sweeps cameras per PHC (1-3), ophthalmologists (1-8) and
%   the AI threshold along the measured ROC (80-97.5 % sensitivity points from
%   config/operating_point.json), plus the no-AI baseline per staffing, with a
%   full simulated year each. Objective: minimum cost subject to missed
%   referable cases <= maxMissedFraction of referable cases (default 20 %) and
%   95th-percentile wait <= maxP95WaitDays (default 7). Returns every run, the
%   Pareto front (cost vs missed) and the four slide numbers.
%
%   opts (struct, all optional): cameras, doctors, maxMissedFraction,
%   maxP95WaitDays, sampleFraction, useParallel (parfor over runs, needs
%   Parallel Computing Toolbox), useSimulink (parsim on district_screening.slx
%   with Simulink.SimulationInput; the block model's doctor pool is
%   DoctorPoolDES.m, so it carries the daily hour budget like the MATLAB DES).
%
%   The result is also written to matlab/simulink/sweep_result.json in the
%   same shape as backend/config/sweep_cache.json so the web front end can
%   display either.

    if nargin < 1, opts = struct(); end
    cameras = getfieldOr(opts, 'cameras', 1:3);
    doctors = getfieldOr(opts, 'doctors', 1:8);
    maxMissedFraction = getfieldOr(opts, 'maxMissedFraction', 0.20);
    maxP95 = getfieldOr(opts, 'maxP95WaitDays', 7);
    sampleFraction = getfieldOr(opts, 'sampleFraction', 1.0);
    useParallel = getfieldOr(opts, 'useParallel', license('test', 'Distrib_Computing_Toolbox'));
    useSimulink = getfieldOr(opts, 'useSimulink', false);

    base = districtParams('sampleFraction', sampleFraction);
    point = drscreen.operatingPoint();
    roc = point.externalTest.roc_points_for_simulation;
    configs = {};
    for cam = cameras
        for doc = doctors
            configs{end+1} = struct('cam', cam, 'doc', doc, 'roc', []); %#ok<AGROW>
            for r = 1:numel(roc)
                configs{end+1} = struct('cam', cam, 'doc', doc, 'roc', roc(r)); %#ok<AGROW>
            end
        end
    end
    n = numel(configs);
    runs = cell(n, 1);
    tStart = tic;
    if useSimulink
        runs = runWithSimulink(configs, base);
    elseif useParallel
        parfor i = 1:n
            runs{i} = summarise(simulateDistrict(paramsFor(base, configs{i})), configs{i});
        end
    else
        for i = 1:n
            runs{i} = summarise(simulateDistrict(paramsFor(base, configs{i})), configs{i});
            if mod(i, 10) == 0, fprintf('  %d/%d\n', i, n); end
        end
    end
    runs = [runs{:}];

    % Pareto front over AI runs: cost ascending, missed strictly decreasing.
    ai = runs([runs.ai]);
    [~, order] = sortrows([[ai.costInrTotal]', [ai.missedTotal]']);
    front = ai([]);
    for k = order'
        if isempty(front) || ai(k).missedTotal < front(end).missedTotal, front(end+1) = ai(k); end %#ok<AGROW>
    end
    referableCases = runs(1).referableCases;
    maxMissed = round(maxMissedFraction * referableCases);
    feasible = @(r) r.waitP95Days <= maxP95 && r.missedTotal <= maxMissed;
    bestAi = cheapest(ai(arrayfun(feasible, ai)));
    baseRuns = runs(~[runs.ai]);
    bestBase = cheapest(baseRuns(arrayfun(feasible, baseRuns)));
    slide = [];
    if ~isempty(bestAi) && ~isempty(bestBase)
        slide = struct('doctorsWithAi', bestAi.ophthalmologists, 'doctorsWithoutAi', bestBase.ophthalmologists, ...
            'costWithAi', bestAi.costInrTotal, 'costWithoutAi', bestBase.costInrTotal, ...
            'missedWithAi', bestAi.missedTotal, 'missedWithoutAi', bestBase.missedTotal, ...
            'operatingPointWithAi', bestAi.sensitivityTarget, 'camerasPerPhc', bestAi.camerasPerPhc);
    end
    sweep = struct('writtenAt', char(datetime('now', 'TimeZone', 'UTC')), 'sampleFraction', sampleFraction, ...
        'constraints', struct('maxMissed', maxMissed, 'maxMissedFraction', maxMissedFraction, ...
            'referableCases', referableCases, 'maxP95WaitDays', maxP95), ...
        'coupledFrom', struct('modelVersion', point.modelVersion, 'fingerprint', point.calibrationFingerprint), ...
        'runs', runs, 'paretoFront', front, 'bestAi', bestAi, 'bestBaseline', bestBase, 'slideNumbers', slide, ...
        'elapsedMs', round(1000 * toc(tStart)), 'engine', ifelse(useSimulink, 'simulink-parsim', 'matlab-des'));
    out = fullfile(fileparts(mfilename('fullpath')), 'sweep_result.json');
    fid = fopen(out, 'w'); fwrite(fid, jsonencode(sweep)); fclose(fid);
    fprintf('wrote %s (%d runs, %.0f s)\n', out, n, sweep.elapsedMs / 1000);
end

function p = paramsFor(base, cfg)
    p = base;
    p.camerasPerPhc = cfg.cam; p.operatorsPerPhc = cfg.cam; p.ophthalmologists = cfg.doc;
    if isempty(cfg.roc)
        p.aiEnabled = false;
    else
        p.aiEnabled = true; p.sensitivity = cfg.roc.sensitivity; p.specificity = cfg.roc.specificity;
    end
end

function s = summarise(r, cfg)
    s = struct('ai', ~isempty(cfg.roc), 'camerasPerPhc', cfg.cam, 'ophthalmologists', cfg.doc, ...
        'sensitivityTarget', NaN, 'sensitivity', NaN, 'specificity', NaN, ...
        'costInrTotal', r.costInrTotal, 'costInrPerPatient', r.costInrPerScreenedPatient, ...
        'referableCases', r.referableCases, 'missedTotal', r.missedTotal, 'missedByAi', r.referableMissedByAi, ...
        'missedByReader', r.referableMissedByReader, 'unresulted', r.referableUnresultedAtYearEnd, ...
        'programmeSensitivity', r.programmeSensitivity, 'unnecessaryReferrals', r.unnecessaryReferrals, ...
        'waitMeanDays', r.waitCaptureToResultDays.mean, 'waitP95Days', r.waitCaptureToResultDays.p95, ...
        'backlogAtEnd', r.backlogAtEnd, 'utilisation', r.ophthalmologistUtilisation, 'doctorHours', r.doctorHours);
    if ~isempty(cfg.roc)
        s.sensitivityTarget = cfg.roc.sensitivity_target; s.sensitivity = cfg.roc.sensitivity; s.specificity = cfg.roc.specificity;
    end
end

function best = cheapest(list)
    best = [];
    if isempty(list), return; end
    [~, i] = min([list.costInrTotal]); best = list(i);
end

function runs = runWithSimulink(configs, base)
    % parsim over Simulink.SimulationInput objects; each run changes only the
    % model-workspace variable `params` (the DoctorPoolDES block parameters
    % are expressions on it), the capture capacity and the generator's
    % Generate action. The terminators' counters (resultCount, dischargeCount,
    % missedCount, logged by To Workspace blocks) come back in the
    % SimulationOutput; the MATLAB DES reference supplies the waiting-time and
    % cost fields the block model does not log, and both missed counts are
    % kept side by side (runs(i).simulink).
    here = fileparts(mfilename('fullpath'));
    addpath(here);
    model = 'district_screening';
    if ~isfile(fullfile(here, [model '.slx'])), buildDistrictModel(base, model); end
    load_system(model);
    n = numel(configs);
    in(1:n) = Simulink.SimulationInput(model);
    for i = 1:n
        p = paramsFor(base, configs{i});
        in(i) = in(i).setVariable('params', p, 'Workspace', model);
        in(i) = in(i).setBlockParameter([model '/Capture + Stage 0'], 'Capacity', num2str(p.phcs * min(p.camerasPerPhc, p.operatorsPerPhc)));
        in(i) = in(i).setBlockParameter([model '/Patient generator'], 'GenerateAction', generateAction(p));
        if getSimulinkBlockHandle([model '/Tele-review + in-person']) > 0
            in(i) = in(i).setBlockParameter([model '/Tele-review + in-person'], 'Capacity', num2str(p.ophthalmologists));
        end
    end
    outs = parsim(in, 'ShowProgress', 'on', 'TransferBaseWorkspaceVariables', 'on', ...
                  'AttachedFiles', {fullfile(here, 'DoctorPoolDES.m'), fullfile(here, 'districtBus.m'), fullfile(here, 'generateAction.m')});
    runs = cell(n, 1);
    for i = 1:n
        r = simulateDistrict(paramsFor(base, configs{i}));
        runs{i} = summarise(r, configs{i});
        runs{i}.simulink = struct('resulted', lastValue(outs(i), 'resultCount'), 'discharged', lastValue(outs(i), 'dischargeCount'), ...
                                  'missedByAi', lastValue(outs(i), 'missedCount'));
    end
end

function v = lastValue(out, name)
    v = 0;                       % a terminator nothing reached logs no variable: zero arrivals
    try
        ts = out.get(name);
        if ~isempty(ts) && ~isempty(ts.Data), v = ts.Data(end); end
    catch
    end
end

function v = getfieldOr(s, name, default)
    if isfield(s, name) && ~isempty(s.(name)), v = s.(name); else, v = default; end
end

function out = ifelse(c, a, b)
    if c, out = a; else, out = b; end
end
