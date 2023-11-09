while read url; do
    wget "$url"
done < links.txt