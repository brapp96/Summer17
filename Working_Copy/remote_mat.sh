echo 'Logging MATLAB script, starting '$(date) > test.log
echo 'matlab -noawt -r "parpool;testfile;quit;"' | sh >> test.log &
