function hex = sha256File(path)
%SHA256FILE  SHA-256 of a file as a lower-case hex string.
%   MATLAB: Java MessageDigest. GNU Octave (no Java in the CLI build): the
%   `hash` builtin, so the fingerprint check in operatingPoint runs there too.
    if exist('OCTAVE_VERSION', 'builtin') == 5
        fid = fopen(path, 'r');
        bytes = fread(fid, inf, '*uint8');
        fclose(fid);
        hex = lower(hash('sha256', char(bytes')));
        return;
    end
    md = java.security.MessageDigest.getInstance('SHA-256');
    fid = fopen(path, 'r');
    bytes = fread(fid, inf, '*uint8');
    fclose(fid);
    md.update(bytes);
    digest = typecast(md.digest(), 'uint8');
    hex = lower(reshape(dec2hex(digest, 2)', 1, []));
end
