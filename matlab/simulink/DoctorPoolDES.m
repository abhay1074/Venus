classdef DoctorPoolDES < matlab.DiscreteEventSystem
%DOCTORPOOLDES  Ophthalmologist pool with a daily hour budget (SimEvents).
%   A MATLAB Discrete-Event System block for district_screening.slx that
%   replaces the plain Entity Server for the doctors. It holds an internal
%   priority queue (AI-positive cases first, the same rule as the Python and
%   MATLAB reference simulations) and NumDoctors server slots. Each slot
%   keeps the minutes it has already worked today; a case that would push a
%   doctor past DailyBudgetMinutes waits until the next working day starts
%   (DayMinutes per day), exactly as startService() in simulateDistrict.m.
%
%   Entities must carry the attributes the Patient generator sets in
%   buildDistrictModel.m: aiPositive (priority key) and reviewMinutes
%   (service time). BusyMinutes accumulates the doctors' working time for the
%   cost model and is readable after the run through the block's discrete
%   state (or the logged signal on the optional output).
%
%   Verified on first MATLAB run: the matlab.DiscreteEventSystem API names
%   below follow the R2024b documentation (queuePriority, serverStorage,
%   eventForward, eventTimer). If a name has moved in your release the block
%   dialog reports it; fix it here and rebuild the model.

    properties (Nontunable)
        NumDoctors = 2;              % parallel ophthalmologists
        DailyBudgetMinutes = 360;    % doctorHoursPerDay * 60
        DayMinutes = 480;            % phcHoursPerDay * 60 (simulation clock is in minutes)
    end

    properties (DiscreteState)
        FreeAt          % 1 x NumDoctors: simulation time each doctor is next free
        UsedToday       % 1 x NumDoctors: minutes worked on the current day
        DayIndex        % 1 x NumDoctors: the day UsedToday refers to
        BusyMinutes     % total minutes of doctor time consumed
        Served          % entities completed
    end

    methods (Access = protected)
        function num = getNumInputsImpl(~)
            num = 1;
        end

        function num = getNumOutputsImpl(~)
            num = 1;
        end

        function entityTypes = getEntityTypesImpl(obj)
            entityTypes = obj.entityType('Patient');
        end

        function [inputTypes, outputTypes] = getEntityPortsImpl(~)
            inputTypes = {'Patient'};
            outputTypes = {'Patient'};
        end

        function [storageSpec, I, O] = getEntityStorageImpl(obj)
            % Storage 1: the waiting room, AI-positive cases first.
            % Storage 2: the doctors (NumDoctors slots).
            storageSpec = [obj.queuePriority('Patient', inf, 'aiPositive', 'descending'), ...
                           obj.serverStorage('Patient', obj.NumDoctors)];
            I = 1;    % input port 1 -> storage 1
            O = 2;    % storage 2 -> output port 1
        end

        function setupImpl(obj)
            obj.FreeAt = zeros(1, obj.NumDoctors);
            obj.UsedToday = zeros(1, obj.NumDoctors);
            obj.DayIndex = zeros(1, obj.NumDoctors);
            obj.BusyMinutes = 0;
            obj.Served = 0;
        end

        function resetImpl(obj)
            obj.setupImpl();
        end

        function [entity, events] = PatientEntry(obj, storage, entity, ~)
            if storage == 1
                % Waiting room: move to a doctor as soon as a slot is free.
                % A forward to a full storage stays pending until it has room.
                events = obj.eventForward('storage', 2, 0);
            else
                % A doctor slot took the case: the same assignment rule as
                % simulateDistrict.startService, with the wait for the next
                % day (when today's budget is spent) folded into the timer.
                now = obj.getCurrentTime();
                service = entity.data.reviewMinutes;
                [~, i] = min(obj.FreeAt);
                t = max(now, obj.FreeAt(i));
                day = floor(t / obj.DayMinutes);
                if day ~= obj.DayIndex(i)
                    obj.DayIndex(i) = day; obj.UsedToday(i) = 0;
                end
                if obj.UsedToday(i) + service > obj.DailyBudgetMinutes
                    day = day + 1; t = day * obj.DayMinutes;
                    obj.DayIndex(i) = day; obj.UsedToday(i) = 0;
                end
                obj.UsedToday(i) = obj.UsedToday(i) + service;
                obj.FreeAt(i) = t + service;
                obj.BusyMinutes = obj.BusyMinutes + service;
                events = obj.eventTimer('done', (t - now) + service);
            end
        end

        function [entity, events] = PatientTimer(obj, ~, entity, ~)
            obj.Served = obj.Served + 1;
            events = obj.eventForward('output', 1, 0);
        end
    end
end
