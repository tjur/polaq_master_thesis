# create new lines
sed -i -e 's/VALUES /VALUES\n/g' $1
sed -i -e 's/),(/),\n(/g' $1

# leave only lines with links to namespace 0
sed -i -e '/^([0-9]\+,0,.*,0)/!d' $1
