#!/usr/bin/env bash

# =====================================================
#  Statistics at the Command Line – Bash Automation
#  This script performs statistical analysis on traffic.csv
# =====================================================

FILE="traffic.csv"

# -----------------------------------------------------
# Create sample dataset
# -----------------------------------------------------
create_data() {
cat > $FILE << EOF
date,visitors,page_views,bounce_rate
2024-01-01,1250,4500,45.2
2024-01-02,1180,4200,47.1
2024-01-03,1520,5800,42.3
2024-01-04,1430,5200,43.8
2024-01-05,980,3400,51.2
2024-01-06,1100,3900,48.5
2024-01-07,1680,6100,40.1
2024-01-08,1550,5600,41.9
2024-01-09,1420,5100,44.2
2024-01-10,1290,4700,46.3
EOF
echo "traffic.csv created."
}

# -----------------------------------------------------
# Basic Exploration
# -----------------------------------------------------

count_rows() {
    wc -l $FILE
}

view_head() {
    head -n 5 $FILE
}

extract_visitors() {
    cut -d',' -f2 $FILE | tail -n +2
}

# -----------------------------------------------------
# Central Tendencies
# -----------------------------------------------------

mean_visitors() {
    cut -d',' -f2 $FILE | tail -n +2 | \
    awk '{sum+=$1; count++} END {print "Mean:", sum/count}'
}

median_visitors() {
    cut -d',' -f2 $FILE | tail -n +2 | sort -n | \
    awk '{arr[NR]=$1; count=NR}
    END {
        if (count%2==1)
            print "Median:", arr[(count+1)/2];
        else
            print "Median:", (arr[count/2]+arr[count/2+1])/2
    }'
}

mode_visitors() {
    cut -d',' -f2 $FILE | tail -n +2 | sort -n | uniq -c | sort -rn | \
    head -n 1 | awk '{print "Mode:", $2, "(appears", $1, "times)"}'
}

# -----------------------------------------------------
# Dispersion Metrics
# -----------------------------------------------------

min_max() {
    awk -F',' '
        NR==2 {min=$2; max=$2}
        NR>2 {if($2<min) min=$2; if($2>max) max=$2}
        END {print "Min:", min, "Max:", max}
    ' $FILE
}

stddev_population() {
    awk -F',' '
        NR>1 {sum+=$2; sumsq+=$2*$2; count++}
        END {
            mean=sum/count;
            print "Std Dev (population):", sqrt((sumsq/count)-(mean*mean))
        }
    ' $FILE
}

stddev_sample() {
    awk -F',' '
        NR>1 {sum+=$2; sumsq+=$2*$2; count++}
        END {
            mean=sum/count;
            print "Std Dev (sample):", sqrt((sumsq-(sum*sum/count))/(count-1))
        }
    ' $FILE
}

variance_population() {
    awk -F',' '
        NR>1 {sum+=$2; sumsq+=$2*$2; count++}
        END {
            mean=sum/count;
            print "Variance:", (sumsq/count)-(mean*mean)
        }
    ' $FILE
}

# -----------------------------------------------------
# Percentiles
# -----------------------------------------------------

quartiles() {
cut -d',' -f2 $FILE | tail -n +2 | sort -n | awk '
{arr[NR]=$1; count=NR}
END {
  q1_pos=(count+1)/4
  q2_pos=(count+1)/2
  q3_pos=3*(count+1)/4

  print "Q1:", arr[int(q1_pos)]
  if (count%2==1)
      print "Median:", arr[int(q2_pos)];
  else
      print "Median:", (arr[count/2]+arr[count/2+1])/2;
  print "Q3:", arr[int(q3_pos)]
}'
}

percentile() {
P=$1
cut -d',' -f2 $FILE | tail -n +2 | sort -n | awk -v p=$P '
{arr[NR]=$1; count=NR}
END {
  pos = (count+1) * p/100
  idx = int(pos)
  frac = pos - idx

  if (idx >= count)
      print p "th percentile:", arr[count]
  else
      print p "th percentile:", arr[idx] + frac * (arr[idx+1] - arr[idx])
}'
}

# -----------------------------------------------------
# Multi-column Averages
# -----------------------------------------------------

multi_averages() {
awk -F',' '
NR>1 {
  v_sum+=$2
  pv_sum+=$3
  br_sum+=$4
  count++
}
END {
  print "Average visitors:", v_sum/count
  print "Average page views:", pv_sum/count
  print "Average bounce rate:", br_sum/count
}' $FILE
}

# -----------------------------------------------------
# Correlation
# -----------------------------------------------------

correlation() {
awk -F', *' '
NR>1 {
  x[NR-1]=$2
  y[NR-1]=$3
  sum_x+=$2
  sum_y+=$3
  count++
}
END {
  if(count<2) exit
  mean_x=sum_x/count
  mean_y=sum_y/count

  for(i=1; i<=count; i++) {
    dx=x[i]-mean_x
    dy=y[i]-mean_y

    cov+=dx*dy
    var_x+=dx*dx
    var_y+=dy*dy
  }

  sd_x=sqrt(var_x/count)
  sd_y=sqrt(var_y/count)
  correlation=(cov/count)/(sd_x*sd_y)
  print "Correlation:", correlation
}' $FILE
}

# -----------------------------------------------------
# Help Menu
# -----------------------------------------------------

help_menu() {
cat <<EOF
Usage: ./stats_cli.sh <command>

Commands:
  create_data         Generate traffic.csv
  count_rows          Count rows in dataset
  view_head           View top 5 rows
  extract_visitors    Show visitors column
  mean                Mean of visitors
  median              Median of visitors
  mode                Mode of visitors
  minmax              Min and Max
  stdpop              Population standard deviation
  stdsample           Sample standard deviation
  variance            Variance
  quartiles           Q1, Median, Q3
  percentile <n>      nth percentile
  averages            Averages for all numeric columns
  correlation         Pearson correlation (visitors vs page_views)
EOF
}

# -----------------------------------------------------
# Command Routing
# -----------------------------------------------------

case "$1" in
  create_data) create_data ;;
  count_rows) count_rows ;;
  view_head) view_head ;;
  extract_visitors) extract_visitors ;;
  mean) mean_visitors ;;
  median) median_visitors ;;
  mode) mode_visitors ;;
  minmax) min_max ;;
  stdpop) stddev_population ;;
  stdsample) stddev_sample ;;
  variance) variance_population ;;
  quartiles) quartiles ;;
  percentile) percentile $2 ;;
  averages) multi_averages ;;
  correlation) correlation ;;
  *) help_menu ;;
esac

