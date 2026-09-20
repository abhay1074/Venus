function p = calibrate(raw, point)
%CALIBRATE  Platt scaling of the raw P(grade >= 2) with the locked (a, b).
%   p = drscreen.calibrate(raw, point) = sigmoid(a * logit(raw) + b)
    raw = min(max(raw, 1e-7), 1 - 1e-7);
    z = point.calibration.a * log(raw / (1 - raw)) + point.calibration.b;
    p = 1 / (1 + exp(-z));
end
