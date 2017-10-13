filename=${1%%.*}
pdflatex ${filename}.tex 
biber ${filename}.bcf > /dev/null 2>&1
pdflatex ${filename}.tex > /dev/null 2>&1
pdflatex ${filename}.tex > /dev/null 2>&1
