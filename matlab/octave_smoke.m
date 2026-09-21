% octave_smoke.m — exercise the pure-logic MATLAB functions under GNU Octave.
% Not a substitute for runTests.m in MATLAB (datetime, unittest, toolboxes are
% absent here), but it executes gradeRule, fuse, calibrate, tier and the
% district simulation with the same cases as the MATLAB tests.
%   octave-cli --no-gui -q octave_smoke.m     (from the matlab/ folder)
pkg load statistics
addpath(pwd); addpath(fullfile(pwd, 'simulink'));
fails = 0;
function r = check(cond, msg)
  if cond, printf('ok    %s\n', msg); r = 0; else, printf('FAIL  %s\n', msg); r = 1; end
end
s1 = @(ma, he, ex, se, heArea, quad, exArea) struct('lesions', struct( ...
  'MA', struct('count', ma, 'areaFraction', 0.001 * (ma > 0), 'centroids', zeros(0, 2)), ...
  'HE', struct('count', he, 'areaFraction', heArea * (he > 0), 'centroids', zeros(0, 2)), ...
  'EX', struct('count', ex, 'areaFraction', exArea * (ex > 0), 'centroids', zeros(0, 2)), ...
  'SE', struct('count', se, 'areaFraction', 0.001 * (se > 0), 'centroids', zeros(0, 2))), ...
  'hemorrhagesPerQuadrant', quad);
g = @(ma, he, ex, se, heArea, quad, exArea) getfield(drscreen.gradeRule(s1(ma, he, ex, se, heArea, quad, exArea), 0), 'grade');
fails += check(g(0,0,0,0,0.001,[0 0 0 0],0.001) == 0, 'rule: no lesions -> 0');
fails += check(g(5,0,0,0,0.001,[0 0 0 0],0.001) == 1, 'rule: MA only -> 1');
fails += check(g(5,3,0,0,0.001,[3 0 0 0],0.001) == 2, 'rule: MA+HE -> 2');
fails += check(g(0,0,4,0,0.001,[0 0 0 0],0.001) == 2, 'rule: EX -> 2');
fails += check(g(0,90,0,0,0.001,[22 25 21 22],0.001) == 3, 'rule: 4-2-1 -> 3');
fails += check(g(0,90,0,0,0.001,[22 25 21 5],0.001) == 2, 'rule: 4-2-1 not met -> 2');
fails += check(getfield(drscreen.gradeRule(s1(0,0,0,0,0.001,[0 0 0 0],0.001), 0.8), 'grade') == 4, 'rule: NV -> 4');
fails += check(g(0,3,0,0,0.06,[3 0 0 0],0.001) == 4, 'rule: HE area -> 4');
out = drscreen.gradeRule(s1(1,0,1,0,0.001,[0 0 0 0],0.00005), 0);
fails += check(out.grade == 0 && any(cellfun(@(x) ~isempty(strfind(x, 'below component threshold')), out.trace)), 'rule: single specks ignored');

point = struct('graderTag', 'test', 'calibration', struct('a', 1, 'b', 0), 'thresholds', struct('referable', 0.5, 'referable85', 0.6, 'abstainBand', 0.05));
p = arrayfun(@(x) drscreen.calibrate(x, point), [1e-6 1e-3 0.01 0.1 0.5 0.9 0.999]);
fails += check(issorted(p) && p(1) > 0 && p(end) < 1, 'calibrate: monotone, bounded');

cnn = struct('grade', 3, 'gradeLabel', 'Severe NPDR', 'referableProbability', 0.95);
rule = struct('grade', 0, 'counts', struct('MA', 0, 'HE', 0, 'EX', 0, 'SE', 0));
f = drscreen.fuse(cnn, rule, point);
fails += check(f.flagForReview && f.disagreementLevels == 3, 'fuse: disagreement flagged');
f2 = drscreen.fuse(struct('grade', 2, 'gradeLabel', 'Moderate NPDR', 'referableProbability', 0.52), struct('grade', 2, 'counts', struct('MA', 3, 'HE', 2, 'EX', 0, 'SE', 0)), point);
fails += check(f2.abstain && f2.flagForReview && f2.referable, 'fuse: abstain band flagged');
f3 = drscreen.fuse(struct('grade', 0, 'gradeLabel', 'No DR', 'referableProbability', 0.05), rule, point);
fails += check(~f3.flagForReview && ~f3.referable, 'fuse: clean negative not flagged');
pointLow = point; pointLow.thresholds.referable = 0.1;
policy = struct('abstainBandLogit', 0.35, 'attentionFloor', 0.2, 'attentionMinLift', NaN, 'chosenOn', 'test');
ruleMod = struct('grade', 2, 'counts', struct('MA', 3, 'HE', 2, 'EX', 0, 'SE', 0));
fHi = drscreen.fuse(struct('grade', 2, 'gradeLabel', 'Moderate NPDR', 'referableProbability', 0.13), ruleMod, pointLow, policy);
fLo = drscreen.fuse(struct('grade', 1, 'gradeLabel', 'Mild NPDR', 'referableProbability', 0.06), ruleMod, pointLow, policy);
fails += check(fHi.abstain && ~fLo.abstain && abs(fHi.abstainLow - 0.0726) < 1e-3 && abs(fHi.abstainHigh - 0.1363) < 1e-3, 'fuse: logit abstain band under policy');
pol = drscreen.reviewPolicy();
fails += check(abs(pol.abstainBandLogit - 0.35) < 1e-9 && abs(pol.attentionFloor - 0.2) < 1e-9 && isnan(pol.attentionMinLift), 'reviewPolicy: reads config/review_policy.json');

res = @(grade, pr, flag, nv) struct('accepted', true, 'stage2', struct('fusion', struct('grade', grade, 'pReferable', pr, 'referable', pr >= 0.334, 'flagForReview', flag, 'flagReasons', {{}}), 'cnn', struct('nvProbability', nv)));
fails += check(strcmp(getfield(drscreen.tier([]), 'tier'), 'P0'), 'tier: reject -> P0');
fails += check(strcmp(getfield(drscreen.tier(res(4, 0.9, false, 0)), 'tier'), 'P1'), 'tier: PDR -> P1');
fails += check(strcmp(getfield(drscreen.tier(res(2, 0.2, false, 0.7)), 'tier'), 'P1'), 'tier: NV -> P1');
fails += check(strcmp(getfield(drscreen.tier(res(2, 0.8, true, 0)), 'tier'), 'P3'), 'tier: flag -> P3');
fails += check(strcmp(getfield(drscreen.tier(res(2, 0.8, false, 0)), 'tier'), 'P2'), 'tier: referable -> P2');
fails += check(strcmp(getfield(drscreen.tier(res(0, 0.05, false, 0)), 'tier'), 'P4'), 'tier: clean -> P4');
fails += check(strcmp(getfield(drscreen.tier(res(0, 0.05, false, 0), struct('pregnant', true)), 'tier'), 'P2'), 'tier: pregnancy raises P4->P2');
t6 = drscreen.tier(res(0, 0.05, false, 0), struct('hba1c', 9.5));
fails += check(strcmp(t6.tier, 'P4') && strcmp(t6.deadlineText, '6-month recall'), 'tier: HbA1c shortens recall');

% District simulation (no operating point needed: pass explicit values)
base = struct('annualPatients', 100000, 'workingDays', 250, 'referablePrevalence', 0.06, 'anyDrPrevalence', 0.18, ...
  'phcs', 30, 'camerasPerPhc', 1, 'operatorsPerPhc', 1, 'captureMinutes', 4, 'retakeProbability', 0.05, 'phcHoursPerDay', 8, ...
  'aiEnabled', true, 'sensitivity', 0.9, 'specificity', 0.6, 'flagRate', 0.1, 'humanReaderSensitivity', 0.85, ...
  'humanReaderSpecificity', 0.9, 'ophthalmologists', 3, 'doctorHoursPerDay', 6, 'teleReviewMinutes', 3, 'inPersonMinutes', 20, ...
  'operatorHourCost', 150, 'doctorHourCost', 1500, 'cameraAnnualCost', 50000, 'seed', 42, 'sampleFraction', 0.1);
base.ophthalmologists = 8;   % unsaturated, so doctor hours reflect the load rather than the capacity
tic; ai = simulateDistrict(base); t = toc;
noai = base; noai.aiEnabled = false; b = simulateDistrict(noai);
printf('      sim 10%% year: %.1f s; AI doctor hours %.0f vs %.0f; missed by AI %d (expected ~%.0f); programme sens %.2f vs %.2f\n', ...
  t, ai.doctorHours, b.doctorHours, ai.referableMissedByAi, ai.referableCases * 0.1 * 0.9, ai.programmeSensitivity, b.programmeSensitivity);
fails += check(ai.doctorHours < b.doctorHours, 'sim: AI reduces doctor hours');
fails += check(b.referableMissedByAi == 0, 'sim: baseline misses nothing at the AI');
expected = ai.referableCases * 0.1 * 0.9;
fails += check(abs(ai.referableMissedByAi - expected) < 0.35 * expected + 20, 'sim: misses follow the confusion matrix');
perfect = base; perfect.sensitivity = 1; perfect.specificity = 1; perfect.flagRate = 0; perfect.sampleFraction = 0.05;
pp = simulateDistrict(perfect);
fails += check(pp.referableMissedByAi == 0 && pp.unnecessaryReferrals == 0, 'sim: perfect AI misses nothing');
few = base; few.ophthalmologists = 2; many = base; many.ophthalmologists = 8;
rf = simulateDistrict(few); rm = simulateDistrict(many);
fails += check(rm.waitCaptureToResultDays.p95 <= rf.waitCaptureToResultDays.p95 && rm.ophthalmologistUtilisation <= 1, 'sim: more doctors, shorter waits');
printf('%d failures\n', fails);
