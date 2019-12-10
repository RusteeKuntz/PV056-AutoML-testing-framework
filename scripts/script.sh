source venv/bin/activate
FILE_DATASET_SPLIT="datasets-split-exp.csv"
FILE_DATASET_RM="datasets-rm-exp.csv"
NOW=""
NOWF=""

now(){
	NOW=$(date +"%d/%m/%Y-%H:%M:%S")
}
nowf(){
  NOWF=$(date +"%d%m%Y-%H%M%S")
}

now
nowf
TIMESTAMP_SPLIT="$NOWF"
printf "%s - Splitting data...\n" "$NOW"
pv056-split-data -c configs/split/default.json -d "$FILE_DATASET_SPLIT" > "logs/log-$TIMESTAMP_SPLIT-split.log"
now
printf "%s - SPLIT done.\n\n" "$NOW"

nowf
TIMESTAMP_OD="$NOWF"
printf "%s - Applying OD methods...\n" "$NOW"
pv056-apply-od-methods -c configs/od/default.json > "logs/log-$TIMESTAMP_OD-od.log"
now
printf "%s - OD done.\n\n" "$NOW"

nowf
TIMESTAMP_RM="$NOWF"
printf "%s - Removing outliers...\n" "$NOW"
pv056-remove-outliers -c configs/rm/default.json -d "$FILE_DATASET_RM" > "logs/log-$TIMESTAMP_RM-rm.log"
now
printf "%s - RM done.\n\n" "$NOW"

nowf
TIMESTAMP_CLF="$NOWF"
printf "%s - Running classification...\n" "$NOW"
pv056-run-clf -c configs/clf/default.json -d "$FILE_DATASET_RM" > "logs/log-$TIMESTAMP_CLF-clf.log"
now
printf "%s - CLF done.\n\n" "$NOW"

nowf
TIMESTAMP_ACC="$NOWF"
printf "%s - Counting accuracy...\n" "$NOW"
pv056-statistics -c configs/stats/default.json > "logs/accurracy-$TIMESTAMP_ACC.csv"
now
printf "%s - ACC done.\n\n" "$NOW"

echo "Script finished."



deactivate
