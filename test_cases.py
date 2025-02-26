# TestA
import timeit
import concurrent.futures
from shopping_cart import Product, Cart


def test_calculate_total():
    cart = Cart()
    product = Product(1, "Widget", 10.0)

    # Populate the cart with 1000 units of the same product.
    for _ in range(1000):
        cart.add_product(product, 1)

    # Measure the execution time of calculate_total() over 1000 iterations.
    total_time = timeit.timeit(lambda: cart.calculate_total(), number=1000)
    print(f"Test A: 'calculate_total()' took {total_time:.6f} seconds over 1000 iterations.")


# TestB
def add_and_remove(cart, product, iterations):
    for _ in range(iterations):
        try:
            cart.add_product(product, 1)
            cart.remove_product(product, 1)
        except ValueError:
            # In a race condition, removal might occur on a non-existent product.

            pass
    # Return the final total from the cart.
    return cart.calculate_total()


def test_cart_operations():
    cart = Cart()
    product = Product(1, "Widget", 10.0)
    iterations_per_thread = 1000
    num_threads = 50  # Simulate 50 concurrent threads

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(add_and_remove, cart, product, iterations_per_thread)
                   for _ in range(num_threads)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    # Ideally, if operations are balanced, the final cart total should be 0.
    print("Test B: Final totals reported by threads:", results)


# -------------------------
# Main Execution
# -------------------------
if __name__ == "__main__":
    print("Running  Test A:")
    test_calculate_total()
    print("\nRunning Test B:")
    test_cart_operations()




