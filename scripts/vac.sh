#!/bin/bash

#pactl list short modules|sed 's/\t/ /g'|grep "module-loopback"

DESCS=()
while true;do
  MODS=($(pactl list short modules|sed 's/\t/ /g'|grep 'module-loopback'|cut -d' ' -f1))

  echo "Current loopbacks: "
  for i in ${!MODS[*]};do

    if [ "${DESCS[$i]}" != "" ];then echo ${MODS[$i]} - ${DESCS[$i]}
    else echo ${MODS[$i]}
    fi
  done

  echo -e "1. List loopbacks\n2. Create loopback\n3. Delete loopback"
  read CHOICE

  case $CHOICE in
    "2")
      echo -n "Successfully created loopback with ID "
      pactl load-module module-loopback latency_msec=1

      echo -n "Enter a description for this loopback: "
      read DESC
      DESCS+=("$DESC")
    ;;
    "3")
      echo "Which one: "
      read MOD

      for i in ${!MODS[*]}; do
        if [ "$MOD" == "${MODS[$i]}" ]; then
          pactl unload-module $MOD
          unset "DESCS[$i]"

		for i in "${!DESCS[@]}"; do
            new_array+=( "${DESCS[i]}" )
          done
          DESCS=("${new_array[@]}")
          unset new_array

          echo -e "Successfully deleted module $MOD\n"
        fi
      done
    ;;
    "4")
      break
    ;;
  esac

  echo
done
