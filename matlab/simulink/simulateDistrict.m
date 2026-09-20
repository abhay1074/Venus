function out = simulateDistrict(p)
%SIMULATEDISTRICT  Discrete-event model of one district screening year.
%   out = simulateDistrict(p) with p from districtParams. Patient entities
%   arrive as a Poisson process across PHCs, wait for a camera, are captured
%   (with a retake loop), pass AI triage (measured sensitivity, specificity,
%   flag rate) or, in the baseline, go straight to the ophthalmologist queue;
%   K doctors with a daily hour budget read them (tele-review, plus in-person
%   time when they call referable). Queues persist across days; anything not
%   resulted by the horizon is the year-end backlog.
%
%   This is the MATLAB reference of the SimEvents model (district_screening.slx
%   built by buildDistrictModel) and of backend/venus/stage4_simulate.py: the
%   same random draws (seeded), the same outputs.

    rng(p.seed, 'twister');
    dayLen = p.phcHoursPerDay * 60;
    days = max(round(p.workingDays * p.sampleFraction), 1);
    n = round(p.annualPatients * p.sampleFraction);
    horizon = days * dayLen;

    arrivals = sort(rand(n, 1) * horizon);
    phcOf = randi(p.phcs, n, 1);
    referable = rand(n, 1) < p.referablePrevalence;
    retake = rand(n, 1) < p.retakeProbability;
    if p.aiEnabled
        aiPositive = false(n, 1);
        aiPositive(referable) = rand(nnz(referable), 1) < p.sensitivity;
        aiPositive(~referable) = rand(nnz(~referable), 1) < (1 - p.specificity);
        flagged = rand(n, 1) < p.flagRate;
        toDoctor = aiPositive | flagged;
    else
        aiPositive = false(n, 1); toDoctor = true(n, 1);
    end
    readerCalls = false(n, 1);
    readerCalls(referable) = rand(nnz(referable), 1) < p.humanReaderSensitivity;
    readerCalls(~referable) = rand(nnz(~referable), 1) < (1 - p.humanReaderSpecificity);

    % Server pools: per-PHC cameras (budget = full day) and doctors (daily budget).
    phcServers = min(p.camerasPerPhc, p.operatorsPerPhc);
    phcFree = zeros(p.phcs, phcServers); phcUsed = zeros(p.phcs, phcServers); phcDay = zeros(p.phcs, phcServers);
    phcBusy = 0;
    K = max(p.ophthalmologists, 1);
    docFree = zeros(1, K); docUsed = zeros(1, K); docDay = zeros(1, K); docBudget = p.doctorHoursPerDay * 60;
    docBusy = 0;

    % Capture is a simple per-PHC FIFO: process arrivals in time order.
    captureDone = nan(n, 1); phcWait = nan(n, 1);
    for i = 1:n
        k = phcOf(i);
        service = p.captureMinutes * (1 + retake(i));
        [phcFree(k, :), phcUsed(k, :), phcDay(k, :), doneT, busy] = startService(phcFree(k, :), phcUsed(k, :), phcDay(k, :), ...
            arrivals(i), service, dayLen, dayLen, horizon);
        phcBusy = phcBusy + busy;
        captureDone(i) = doneT; phcWait(i) = doneT - service - arrivals(i);
    end

    % Doctor queue, event-driven: cases become ready at capture time; whenever
    % a doctor is free, the highest-priority WAITING case is served (AI-positive
    % before review-only flags, FIFO within a class) -- the same dispatch as the
    % Python event loop, not a global sort. Two FIFO lists suffice because
    % captures are processed in time order.
    resultTime = captureDone; resultTime(toDoctor) = NaN;
    docWait = nan(n, 1);
    [~, byCapture] = sort(captureDone);
    q0 = zeros(n, 1); h0 = 1; t0 = 0;      % AI-positive queue (head, tail)
    q1 = zeros(n, 1); h1 = 1; t1 = 0;      % review-only queue
    for j = 1:n
        i = byCapture(j);
        now = captureDone(i);
        % Serve everything that can start before this capture arrives.
        [q0, h0, q1, h1, docFree, docUsed, docDay, docBusy, resultTime, docWait] = ...
            dispatch(now, q0, h0, t0, q1, h1, t1, docFree, docUsed, docDay, docBusy, resultTime, docWait, ...
                     captureDone, readerCalls, p, docBudget, dayLen, horizon);
        if toDoctor(i)
            if aiPositive(i), t0 = t0 + 1; q0(t0) = i; else, t1 = t1 + 1; q1(t1) = i; end
        end
    end
    [q0, h0, q1, h1, docFree, docUsed, docDay, docBusy, resultTime, docWait] = ...
        dispatch(inf, q0, h0, t0, q1, h1, t1, docFree, docUsed, docDay, docBusy, resultTime, docWait, ...
                 captureDone, readerCalls, p, docBudget, dayLen, horizon); %#ok<ASGLU>

    done = ~isnan(resultTime) & resultTime <= horizon;
    waitDays = (resultTime(done) - captureDone(done)) / dayLen;
    seen = toDoctor & done;
    missedByAi = referable & ~toDoctor;
    missedByReader = referable & seen & ~readerCalls;
    unresulted = referable & toDoctor & ~done;
    unnecessary = toDoctor & ~referable;
    confirmed = referable & seen & readerCalls;
    scale = 1 / p.sampleFraction;
    operatorHours = phcBusy / 60; doctorHours = docBusy / 60;
    doctorCapacity = p.ophthalmologists * p.doctorHoursPerDay * days;
    cameraCost = p.cameraAnnualCost * p.phcs * p.camerasPerPhc * (days / p.workingDays);
    cost = operatorHours * p.operatorHourCost + doctorHours * p.doctorHourCost + cameraCost;

    out = struct('params', p, 'simulated', struct('patients', n, 'days', days), ...
        'waitCaptureToResultDays', struct('mean', mean(waitDays), 'p95', prctile(waitDays, 95), 'max', max([waitDays; 0])), ...
        'phcWaitMinutesMean', mean(phcWait, 'omitnan'), 'doctorWaitDaysMean', mean(docWait, 'omitnan') / dayLen, ...
        'ophthalmologistUtilisation', doctorHours / max(doctorCapacity, 1e-6), ...
        'backlogAtEnd', round(nnz(~done) * scale), 'sentToOphthalmologist', round(nnz(toDoctor) * scale), ...
        'sentFraction', mean(toDoctor), 'referableCases', round(nnz(referable) * scale), ...
        'referableMissedByAi', round(nnz(missedByAi) * scale), 'referableMissedByReader', round(nnz(missedByReader) * scale), ...
        'referableUnresultedAtYearEnd', round(nnz(unresulted) * scale), 'referableConfirmed', round(nnz(confirmed) * scale), ...
        'programmeSensitivity', nnz(confirmed) / max(nnz(referable), 1), 'unnecessaryReferrals', round(nnz(unnecessary) * scale), ...
        'operatorHours', operatorHours * scale, 'doctorHours', doctorHours * scale, ...
        'costInrTotal', round(cost * scale), 'costInrPerScreenedPatient', cost / max(n, 1), ...
        'patientsPerOphthalmologistHour', n / max(doctorHours, 1e-6));
    out.missedTotal = out.referableMissedByAi + out.referableMissedByReader + out.referableUnresultedAtYearEnd;
end

function [q0, h0, q1, h1, docFree, docUsed, docDay, docBusy, resultTime, docWait] = dispatch(now, q0, h0, t0, q1, h1, t1, ...
        docFree, docUsed, docDay, docBusy, resultTime, docWait, captureDone, readerCalls, p, docBudget, dayLen, horizon)
    % Start queued cases while the earliest-free doctor is free before `now`.
    while h0 <= t0 || h1 <= t1
        [freeT, ~] = min(docFree);
        if freeT > now, break; end
        if h0 <= t0, i = q0(h0); h0 = h0 + 1; else, i = q1(h1); h1 = h1 + 1; end
        start = max(freeT, captureDone(i));
        service = p.teleReviewMinutes + p.inPersonMinutes * readerCalls(i);
        [docFree, docUsed, docDay, doneT, busy] = startService(docFree, docUsed, docDay, start, service, docBudget, dayLen, horizon);
        docBusy = docBusy + busy;
        resultTime(i) = doneT; docWait(i) = doneT - captureDone(i);
    end
end

function [freeAt, used, today, doneT, busy] = startService(freeAt, used, today, now, service, budget, dayLen, horizon)
    [~, i] = min(freeAt);
    t = max(now, freeAt(i));
    day = floor(t / dayLen);
    if day ~= today(i), today(i) = day; used(i) = 0; end
    if used(i) + service > budget
        day = day + 1; t = day * dayLen; today(i) = day; used(i) = 0;
    end
    used(i) = used(i) + service;
    freeAt(i) = t + service;
    busy = 0;
    if t < horizon, busy = min(service, horizon - t); end
    doneT = t + service;
end
