classdef GradeTest < matlab.unittest.TestCase
    %GRADETEST  Stage 2 rule grader, fusion and calibration (no networks needed).

    methods (Static)
        function s1 = stage1(ma, he, ex, se, heArea, quadrants, exArea)
            if nargin < 5 || isempty(heArea), heArea = 0.001; end
            if nargin < 6 || isempty(quadrants), quadrants = [he 0 0 0]; end
            if nargin < 7 || isempty(exArea), exArea = 0.001; end
            f = @(count, area) struct('count', count, 'areaFraction', area * (count > 0), 'centroids', zeros(0, 2));
            s1.lesions = struct('MA', f(ma, 0.001), 'HE', f(he, heArea), 'EX', f(ex, exArea), 'SE', f(se, 0.001));
            s1.hemorrhagesPerQuadrant = quadrants;
        end
        function point = fakePoint()
            point = struct('graderTag', 'test', 'calibration', struct('a', 1, 'b', 0), ...
                'thresholds', struct('referable', 0.5, 'referable85', 0.6, 'abstainBand', 0.05));
        end
    end

    methods (Test)
        function ruleFollowsIcdrTable(t)
            s = @GradeTest.stage1;
            g = @(stage1, nv) getfield(drscreen.gradeRule(stage1, nv), 'grade');
            t.verifyEqual(g(s(0, 0, 0, 0), 0), 0);
            t.verifyEqual(g(s(5, 0, 0, 0), 0), 1);
            t.verifyEqual(g(s(5, 3, 0, 0), 0), 2);
            t.verifyEqual(g(s(0, 0, 4, 0), 0), 2);
            t.verifyEqual(g(s(0, 90, 0, 0, [], [22 25 21 22]), 0), 3);
            t.verifyEqual(g(s(0, 90, 0, 0, [], [22 25 21 5]), 0), 2);
            t.verifyEqual(g(s(0, 0, 0, 0), 0.8), 4);
            t.verifyEqual(g(s(0, 3, 0, 0, 0.06), 0), 4);
        end

        function componentThresholdsIgnoreSingleSpecks(t)
            out = drscreen.gradeRule(GradeTest.stage1(1, 0, 1, 0, [], [], 0.00005), 0);
            t.verifyEqual(out.grade, 0);
            t.verifyTrue(any(contains(out.trace, 'below component threshold')));
        end

        function calibrationIsMonotoneAndBounded(t)
            point = GradeTest.fakePoint();
            p = arrayfun(@(x) drscreen.calibrate(x, point), [1e-6 1e-3 0.01 0.1 0.5 0.9 0.999]);
            t.verifyTrue(issorted(p));
            t.verifyGreaterThan(p(1), 0); t.verifyLessThan(p(end), 1);
        end

        function fusionFlagsDisagreementAndAbstain(t)
            point = GradeTest.fakePoint();
            cnn = struct('grade', 3, 'gradeLabel', 'Severe NPDR', 'referableProbability', 0.95);
            rule = struct('grade', 0, 'counts', struct('MA', 0, 'HE', 0, 'EX', 0, 'SE', 0));
            f = drscreen.fuse(cnn, rule, point);
            t.verifyTrue(f.flagForReview); t.verifyEqual(f.disagreementLevels, 3);
            t.verifyTrue(any(contains(f.flagReasons, 'no lesion')));
            cnn2 = struct('grade', 2, 'gradeLabel', 'Moderate NPDR', 'referableProbability', 0.52);
            rule2 = struct('grade', 2, 'counts', struct('MA', 3, 'HE', 2, 'EX', 0, 'SE', 0));
            f2 = drscreen.fuse(cnn2, rule2, point);
            t.verifyTrue(f2.abstain); t.verifyTrue(f2.flagForReview); t.verifyTrue(f2.referable);
            cnn3 = struct('grade', 0, 'gradeLabel', 'No DR', 'referableProbability', 0.05);
            f3 = drscreen.fuse(cnn3, rule, point);
            t.verifyFalse(f3.flagForReview); t.verifyFalse(f3.referable);
        end

        function operatingPointFingerprintIsVerified(t)
            % Reads the real config; must load without error and match the
            % manifest on disk. (Skips if the repo has no operating point yet.)
            cfg = fullfile(drscreen.repoRoot(), 'backend', 'config', 'operating_point.json');
            t.assumeTrue(isfile(cfg), 'no operating_point.json in this checkout');
            point = drscreen.operatingPoint();
            t.verifyGreaterThan(point.thresholds.referable, 0);
            t.verifyLessThan(point.thresholds.referable, 1);
        end
    end
end
