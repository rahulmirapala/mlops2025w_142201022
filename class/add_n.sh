read -p "Enter the number: " n
if [ "$n" -lt 0 ]; then
    echo "number should be greater than zero"
    exit 1
fi
num=$n
total=0
while [ "$num" -gt 0 ]; do
    total=$((total+num))
    num=$((num-1))
done

echo "The sum of first $n natural numbers is $total"
    