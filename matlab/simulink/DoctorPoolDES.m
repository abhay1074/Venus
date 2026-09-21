classdef DoctorPoolDES < matlab.DiscreteEventSystem
%DOCTORPOOLDES  Ophthalmologist pool with a daily hour budget (SimEvents).
%   A MATLAB Discrete-Event System block for district_screening.slx that
%   replaces the plain Entity Server for the doctors: NumDoctors slots, each
%   keeping the minutes it has already worked today. A case that would push
%   a doctor past DailyBudgetMinutes waits until the next working day starts
%   (DayMinutes per day) before its service begins - exactly startService()
%   in simulateDistrict.m. The waiting room in front of it is an ordinary
%   Entity Queue block (priority on aiPositive, AI-positive cases first), so
%   blocking when every slot is busy is handled by the engine, as in
%   MathWorks' desCustomServer pattern.
%
%   Entities must carry reviewMinutes (service time, set by the Patient
%   generator's Generate action). BusyMinutes accumulates the doctors'
%   working time for the cost model.
%
%   Verified in R2026a: one bounded storage (queueFIFO of NumDoctors slots)
%   and eventForward('output', 1, delay) with the whole delay (wait for the
%   next day + service) on the forward; entity event methods are public;
%   discrete-state widths are declared in getDiscreteStateSpecificationImpl.

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
            % The doctors: NumDoctors slots. Full = the upstream queue holds.
            storageSpec = obj.queueFIFO('Patient', obj.NumDoctors);
            I = 1;
            O = 1;
        end

        function [sz, dt, cp] = getDiscreteStateSpecificationImpl(obj, name)
            % Per-doctor vectors are sized by the block parameter (code
            % generation needs the widths declared, not inferred).
            switch name
                case {'FreeAt', 'UsedToday', 'DayIndex'}, sz = [1 obj.NumDoctors];
                otherwise, sz = [1 1];
            end
            dt = 'double'; cp = false;
        end

        function setupImpl(obj)
            obj.FreeAt = zeros(1, obj.NumDoctors);
            obj.UsedToday = zeros(1, obj.NumDoctors);
            obj.DayIndex = zeros(1, obj.NumDoctors);
            obj.BusyMinutes = 0;
            obj.Served = 0;
        end

        function resetImpl(obj)
            obj.FreeAt = zeros(1, obj.NumDoctors);
            obj.UsedToday = zeros(1, obj.NumDoctors);
            obj.DayIndex = zeros(1, obj.NumDoctors);
            obj.BusyMinutes = 0;
            obj.Served = 0;
        end
    end

    methods
        % Entity event methods are called by the SimEvents engine: public.
        function [entity, events] = PatientEntry(obj, ~, entity, ~)
            % A doctor slot took the case: the same assignment rule as
            % simulateDistrict.startService, with the wait for the next day
            % (when today's budget is spent) folded into the forward delay.
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
            obj.Served = obj.Served + 1;
            events = obj.eventForward('output', 1, (t - now) + service);
        end
    end
end
