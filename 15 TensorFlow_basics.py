import tensorflow as tk

X1 = tk.constant(5)
x2 = tk.constant(6)

result = tk.multiply(X1, x2)
print(result)

# creates a new context in which we can run the operation
sess = tk.Session()
print(sess.run(result))
sess.close()

# or

with tk.Session() as sess:
    out = sess.run(result)
    print(out)
