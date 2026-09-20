classdef SimulateTest < matlab.unittest.TestCase
    %SIMULATETEST  Stage 4 district simulation (MATLAB DES reference).

    methods (TestClassSetup)
        function addPaths(t) %#ok<MANU>
            addpath(fullfile(drscreen.repoRoot(), 'matlab', 'simulink'));
        end
    end

    methods (Test)
        function aiReducesDoctorLoadAndMissesComeFromConfusionMatrix(t)
            p = districtParams('sampleFraction', 0.1, 'aiEnabled', true, 'sensitivity', 0.9, 'specificity', 0.6, 'flagRate', 0.1);
            ai = simulateDistrict(p);
            base = simulateDistrict(districtParams('sampleFraction', 0.1, 'aiEnabled', false));
            t.verifyLessThan(ai.doctorHours, base.doctorHours);
            t.verifyEqual(base.referableMissedByAi, 0);
            expected = ai.referableCases * (1 - 0.9) * (1 - 0.1);
            t.verifyLessThan(abs(ai.referableMissedByAi - expected), 0.35 * expected + 20);
        end

        function perfectAiMissesNothing(t)
            out = simulateDistrict(districtParams('sampleFraction', 0.05, 'sensitivity', 1, 'specificity', 1, 'flagRate', 0));
            t.verifyEqual(out.referableMissedByAi, 0); t.verifyEqual(out.unnecessaryReferrals, 0);
        end

        function moreDoctorsShortenWaits(t)
            few = simulateDistrict(districtParams('sampleFraction', 0.1, 'ophthalmologists', 2));
            many = simulateDistrict(districtParams('sampleFraction', 0.1, 'ophthalmologists', 8));
            t.verifyLessThanOrEqual(many.waitCaptureToResultDays.p95, few.waitCaptureToResultDays.p95);
            t.verifyLessThanOrEqual(many.ophthalmologistUtilisation, 1);
            t.verifyLessThanOrEqual(few.ophthalmologistUtilisation, 1);
        end

        function sweepFrontIsMonotone(t)
            sweep = runSweep(struct('cameras', 1, 'doctors', [2 4], 'sampleFraction', 0.04, 'useParallel', false));
            front = sweep.paretoFront;
            t.verifyTrue(issorted([front.costInrTotal]));
            t.verifyTrue(issorted(-[front.missedTotal]));
        end
    end
end
