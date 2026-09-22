function hex = sha256File(path)
%SHA256FILE  SHA-256 of a text artefact as a lower-case hex string.
%   Line endings are normalised to LF before hashing (a Windows checkout with
%   autocrlf must not break the fingerprint check), exactly as
%   backend.venus.config.sha256_of_file does. MATLAB: Java MessageDigest;
%   GNU Octave (no Java in the CLI build): the `hash` builtin.
    fid = fopen(path, 'r');
    bytes = fread(fid, inf, '*uint8')';
    fclose(fid);
    text = strrep(char(bytes), [char(13) char(10)], char(10));
    if exist('OCTAVE_VERSION', 'builtin') == 5
        hex = lower(hash('sha256', text));
        return;
    end
    md = java.security.MessageDigest.getInstance('SHA-256');
    md.update(uint8(text));
    digest = typecast(md.digest(), 'uint8');
    hex = lower(reshape(dec2hex(digest, 2)', 1, []));
end
