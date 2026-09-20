classdef ScheduleTest < matlab.unittest.TestCase
    %SCHEDULETEST  Stage 5 tiers and the allocator (ageing, bumping).

    methods (Static)
        function r = result(grade, p, flag, nv)
            if nargin < 3, flag = false; end
            if nargin < 4, nv = 0; end
            reasons = {}; if flag, reasons = {'x'}; end
            r.accepted = true;
            r.stage2.fusion = struct('grade', grade, 'pReferable', p, 'referable', p >= 0.334, ...
                'flagForReview', flag, 'flagReasons', {reasons});
            r.stage2.cnn = struct('nvProbability', nv);
        end
        function slots = slots(now, facilityType, perDay, ndays)
            slots = struct('id', {}, 'facilityId', {}, 'startsAt', {}, 'booked', {}, 'appointmentId', {}, 'holder', {});
            for d = 1:ndays
                for k = 0:perDay - 1
                    slots(end+1) = struct('id', sprintf('%s-%d-%d', facilityType, d, k), 'facilityId', facilityType, ...
                        'startsAt', now + days(d) + hours(k), 'booked', false, 'appointmentId', '', 'holder', []); %#ok<AGROW>
                end
            end
        end
    end

    methods (Test)
        function tiersFollowTheTable(t)
            r = @ScheduleTest.result;
            t.verifyEqual(drscreen.tier([]).tier, 'P0');
            t.verifyEqual(drscreen.tier(r(4, 0.9)).tier, 'P1');
            t.verifyEqual(drscreen.tier(r(2, 0.2, false, 0.7)).tier, 'P1');
            t.verifyEqual(drscreen.tier(r(3, 0.9), struct('symptoms', {{'floaters'}})).tier, 'P1');
            t.verifyEqual(drscreen.tier(r(2, 0.8, true)).tier, 'P3');
            t.verifyEqual(drscreen.tier(r(2, 0.8)).tier, 'P2');
            t.verifyEqual(drscreen.tier(r(0, 0.05)).tier, 'P4');
        end

        function riskFactorsRaiseByOneNeverLower(t)
            r = @ScheduleTest.result;
            t.verifyEqual(drscreen.tier(r(0, 0.05), struct('pregnant', true)).tier, 'P2');
            t.verifyEqual(drscreen.tier(r(2, 0.8), struct('symptoms', {{'blurred_vision'}})).tier, 'P1');
            routine = drscreen.tier(r(0, 0.05), struct('hba1c', 9.5));
            t.verifyEqual(routine.tier, 'P4'); t.verifyEqual(routine.deadlineText, '6-month recall');
            t.verifyEqual(drscreen.tier(r(4, 0.99), struct('hba1c', 5)).tier, 'P1');
        end

        function allocatorAgesLowerTiers(t)
            now = datetime(2026, 9, 20, 9, 0, 0);
            facilities = containers.Map({'clinic'}, {struct('type', 'clinic', 'pos', [0 0])});
            slots = ScheduleTest.slots(now, 'clinic', 1, 40);
            queue = struct('id', {'A', 'B'}, 'tier', {'P2', 'P3'}, 'riskScore', {0.9, 0.4}, ...
                'queuedAt', {now, now - days(5)}, 'deadlineDays', {30, 3}, 'villagePos', {[0 0], [0 0]});
            decisions = drscreen.allocate(queue, slots, now, facilities);
            byId = containers.Map({decisions.appointmentId}, num2cell(1:numel(decisions)));
            % The overdue P3 takes the first slot even though P2 has a higher risk score.
            t.verifyTrue(decisions(byId('B')).startsAt < decisions(byId('A')).startsAt);
        end

        function allocatorBumpsLowerPriorityWhenDeadlineFull(t)
            now = datetime(2026, 9, 20, 9, 0, 0);
            facilities = containers.Map({'district_hospital'}, {struct('type', 'district_hospital', 'pos', [0 0])});
            slots = ScheduleTest.slots(now, 'district_hospital', 1, 40);
            for d = 1:7
                slots(d).booked = true; slots(d).appointmentId = sprintf('H%d', d);
                slots(d).holder = struct('id', sprintf('H%d', d), 'tier', 'P2', 'queuedAt', now, 'deadlineDays', 30);
            end
            urgent = struct('id', 'U', 'tier', 'P1', 'riskScore', 0.99, 'queuedAt', now, 'deadlineDays', 7, 'villagePos', [0 0]);
            d = drscreen.allocate(urgent, slots, now, facilities);
            t.verifyNotEmpty(d(1).slotId); t.verifyNotEmpty(d(1).bumped);
            t.verifyTrue(d(1).startsAt <= now + days(7));
            t.verifyTrue(d(1).bumped.startsAt > d(1).startsAt);
        end
    end
end
