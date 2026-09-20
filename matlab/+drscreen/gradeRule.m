function rule = gradeRule(stage1, nvProbability)
%GRADERULE  ICDR scale applied literally to the Stage 1 lesion counts.
%   rule = drscreen.gradeRule(stage1, nvProbability) returns a struct with
%   fields grade (0-4), gradeLabel, referable, counts, rawCounts,
%   hemorrhagesPerQuadrant, nvProbability, trace (cellstr) and criteriaText.
%
%   A clinician can check every line of `trace` against the ICDR table:
%     0  no MA, HE or EX above component thresholds
%     1  microaneurysms only
%     2  more than MA only, less than severe
%     3  4-2-1 rule: > 20 hemorrhages in each of 4 quadrants (venous beading
%        and IRMA are not detected and are stated as such)
%     4  P(NV) > 0.5 from the grader, or hemorrhage area suggesting a
%        vitreous / preretinal bleed
%
%   Component thresholds mirror backend/venus/stage2_grade.py EVIDENCE_FLOOR:
%   a lesion type counts only above a minimum component count and total area.

    labels = drscreen.icdrLabels();
    floorMA = [2, 0.0];  floorHE = [1, 0.00025];  floorEX = [3, 0.00020];  floorSE = [1, 0.00040];
    floors = struct('MA', floorMA, 'HE', floorHE, 'EX', floorEX, 'SE', floorSE);
    severeHEPerQuadrant = 20;
    pdrHEAreaFraction   = 0.04;

    keys = {'MA', 'HE', 'EX', 'SE'};
    rawCounts = struct(); present = struct();
    for i = 1:numel(keys)
        k = keys{i};
        rawCounts.(k) = stage1.lesions.(k).count;
        ok = stage1.lesions.(k).count >= floors.(k)(1) && stage1.lesions.(k).areaFraction >= floors.(k)(2);
        if ok
            present.(k) = stage1.lesions.(k).count;
        else
            present.(k) = 0;
        end
    end
    ma = present.MA; he = present.HE; ex = present.EX; se = present.SE;
    heArea = stage1.lesions.HE.areaFraction;
    quadrants = stage1.hemorrhagesPerQuadrant;
    trace = {};

    if nvProbability > 0.5
        grade = 4;
        trace{end+1} = sprintf('P(NV) = %.2f > 0.5 from the grader (classifier, not localised) -> PDR', nvProbability);
    elseif heArea > pdrHEAreaFraction
        grade = 4;
        trace{end+1} = sprintf('hemorrhage area %.1f%% of FOV suggests vitreous/preretinal bleed -> PDR', 100 * heArea);
    elseif all(quadrants > severeHEPerQuadrant)
        grade = 3;
        trace{end+1} = sprintf('4-2-1 rule: >%d hemorrhages in each of 4 quadrants (%s) -> severe NPDR', ...
            severeHEPerQuadrant, mat2str(quadrants));
    elseif he > 0 || ex > 0 || se > 0
        grade = 2;
        parts = {};
        if he > 0, parts{end+1} = sprintf('%d hemorrhage(s) (quadrants %s)', he, mat2str(quadrants)); end
        if ex > 0, parts{end+1} = sprintf('%d hard exudate(s)', ex); end
        if se > 0, parts{end+1} = sprintf('%d soft exudate(s)', se); end
        if ma > 0, parts{end+1} = sprintf('%d microaneurysm(s)', ma); end
        trace{end+1} = ['more than microaneurysms only, less than severe: ' strjoin(parts, '; ') ' -> moderate NPDR'];
    elseif ma > 0
        grade = 1;
        trace{end+1} = sprintf('microaneurysms only (%d) -> mild NPDR', ma);
    else
        grade = 0;
        trace{end+1} = 'no MA, HE or EX above component thresholds -> no DR';
    end

    below = {};
    for i = 1:numel(keys)
        k = keys{i};
        if rawCounts.(k) > 0 && present.(k) == 0
            below{end+1} = sprintf('%s %d', k, rawCounts.(k)); %#ok<AGROW>
        end
    end
    if ~isempty(below)
        trace{end+1} = ['below component threshold, not counted: ' strjoin(below, ', ')];
    end
    trace{end+1} = 'venous beading and IRMA are not detected by this build and are not part of the severe criterion';

    rule = struct('grade', grade, 'gradeLabel', labels{grade + 1}, 'referable', grade >= 2, ...
        'counts', present, 'rawCounts', rawCounts, 'hemorrhagesPerQuadrant', quadrants, ...
        'nvProbability', nvProbability, 'trace', {trace}, 'criteriaText', trace{1});
end
