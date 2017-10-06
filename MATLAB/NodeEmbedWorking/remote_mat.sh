echo 'Logging MATLAB script, starting '$(date) > test2.log
echo 'matlab -noawt -r "PPItest;quit;"' | sh >> test2.log &
