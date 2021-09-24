#!/bin/bash
# Fetches class data from peoplesoft, using the mobile site API

COOKIE_FILE=/tmp/cookie
DBURL="https://highpoint-prd.ps.umbc.edu/app/catalog/getClassSearch"
SCRATCH=/tmp/out.html
TERM="2198"

# Need to go to search page first to get cookie(s), then able to search
echo "Fetching cookies.."
curl -b $COOKIE_FILE -c $COOKIE_FILE -L "https://highpoint-prd.ps.umbc.edu/app/catalog/classSearch" &> /dev/null
[[ $? != 0 ]] && exit

token=$(grep CSRF $COOKIE_FILE|cut -d'	' -f7)

# List of subject inteded for eventual iteration (to get ALL classes)
SUBJECTS=(
ACTG
AFST
)

SUBJECT="CMSC"

echo "Fetching classes.."
curl -b $COOKIE_FILE -c $COOKIE_FILE -L -o "$SCRATCH" -d "CSRFToken=$token&term=$TERM&subject=$SUBJECT" $DBURL 2>/dev/null

sed -i 's/^  *//g' "$SCRATCH" # Remove leading whitespaces to shrink/clean output
tail -c+33 "$SCRATCH" > /tmp/tmp # Remove excess HTML at beginning (may not be needed)
mv /tmp/tmp "$SCRATCH"

echo "Mining data.."
headers=$(grep "class-title-header" $SCRATCH)

# Extract all class IDs
IDs=($(echo $headers|grep -o 'id=[A-Z]\+[0-9]\+'|cut -d'=' -f2))
numIDs=${#IDs[@]}

# Extract titles
IFS=$'\n'
titles=($(echo $headers|grep -o "$SUBJECT [A-Za-z0-9 -]\+"))
unset IFS

# Print classes
for i in $(seq 0 $((numIDs-1)));do
	numSections=$(grep ${IDs[$i]} $SCRATCH|wc -l)
	numSections=$((numSections-1)) # Header line contains the ID as well

	echo "${titles[$i]} (${IDs[$i]}) has $numSections sections"
done

# Get last line in each section
lines=($(cat $SCRATCH |grep -n \</a\>|cut -d':' -f1))

# Using the last line, get all lines in section
IFS=$'\n'
for end in ${lines[@]};do
	start=$((end-17))

	content=$(sed -n "$start,$end p" "$SCRATCH") # All lines for section

	# Section, session, days/time, instructor, status are in this var
	info=$(printf '%s\n' $content|grep section-body|sed -e 's/<.*[ "]>//g' -e 's/<.*>//')
done
unset IFS
