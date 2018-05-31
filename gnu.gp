set term postscript eps color
#set term png
set logscale y
set datafile separator " "
set output 'output.png'
set ylabel 'loss'
set xlabel 'iterations'
plot 'error.txt' using 1 smooth bezier