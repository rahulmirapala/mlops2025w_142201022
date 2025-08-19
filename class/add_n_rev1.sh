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

num1=$n
prod=1
while [ "$num1" -gt 0 ]; do
    prod=$((prod*num1))
    num1=$((num1-1))
done

echo "The product of first $n natural numbers is $prod"
    