function hex = sha256File(path)
%SHA256FILE  SHA-256 of a file as a lower-case hex string (Java MessageDigest).
    md = java.security.MessageDigest.getInstance('SHA-256');
    fid = fopen(path, 'r');
    bytes = fread(fid, inf, '*uint8');
    fclose(fid);
    md.update(bytes);
    digest = typecast(md.digest(), 'uint8');
    hex = lower(reshape(dec2hex(digest, 2)', 1, []));
end
