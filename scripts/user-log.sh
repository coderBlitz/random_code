#!/bin/bash

users=($(users));
# echo ${users[*]};
# () defines an array in bash
# ${#array[*]} returns array length. Replacing * with number N will give length of string at index N

LOG_FILE="/root/users.log.gz"
DATE="date +'[%d/%m/%y %H:%M:%S]'"

while true;do
  tmp=($(users));

# These 3 lines probably are not necessary anymore
#  if [ ${#users[*]} == 0 ] && [ ${#tmp[*]} == 1 ];then
#    echo "$(eval $DATE) \"${tmp[$i]}\" logged in."|tee|gzip -c >> $LOG_FILE;
#    users=(${tmp[*]});

  if [ "${users[*]}" != "${tmp[*]}" ];then
#    echo "User logged in/out";
    echo -e "Users: ${#users[*]}\ntmp: ${#tmp[*]}"

#   If the user logged out
    if ((${#users[*]} > ${#tmp[*]}));then
#      echo "User logged out";
      for i in $(seq 0 $((${#users[*]}-1)) );do
	if [ "${users[$i]}" != "${tmp[$i]}" ];then
	  echo "$(eval $DATE) \"${users[$i]}\" logged out."|tee|gzip -c >> $LOG_FILE;
	  break;
	fi;
      done

#   Else the user has logged in
    else
#      echo "User logged in";
      for i in $(seq 0 $((${#tmp[*]}-1)) );do
	if [ "${users[$i]}" != "${tmp[$i]}" ];then
	  echo "$(eval $DATE) \"${tmp[$i]}\" logged in."|tee|gzip -c >> $LOG_FILE;
	  break;
	fi;
      done
    fi
# Update current users array
    users=(${tmp[*]});
  else
#    echo "No new user";
    sleep 1;
  fi

done
