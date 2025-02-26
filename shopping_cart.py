# shopping_cart.py

class Product:
    def __init__(self, product_id, name, price):
        self.product_id = product_id
        self.name = name
        self.price = price

    def __repr__(self):
        return f"Product({self.product_id}, {self.name}, {self.price})"


class Cart:
    def __init__(self):
        # Dictionary mapping product_id to a tuple (Product, quantity)
        self.items = {}

    def add_product(self, product, quantity=1):
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
        if product.product_id in self.items:
            current_qty = self.items[product.product_id][1]
            self.items[product.product_id] = (product, current_qty + quantity)
        else:
            self.items[product.product_id] = (product, quantity)

    def remove_product(self, product, quantity=1):
        if product.product_id not in self.items:
            raise ValueError("Product not in cart")
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
        current_qty = self.items[product.product_id][1]
        if quantity >= current_qty:
            del self.items[product.product_id]
        else:
            self.items[product.product_id] = (product, current_qty - quantity)

    def calculate_total(self):
        total = 0.0
        for product, quantity in self.items.values():
            total += product.price * quantity
        return total

    def apply_discount(self, discount_rate):
        if discount_rate < 0 or discount_rate > 100:
            raise ValueError("Discount rate must be between 0 and 100")
        total = self.calculate_total()
        discount_amount = total * (discount_rate / 100.0)
        return total - discount_amount


class Order:
    def __init__(self, cart, customer_name):
        self.cart = cart
        self.customer_name = customer_name
        self.total_amount = cart.calculate_total()
        self.status = "Pending"

    def process_order(self):
        if self.total_amount <= 0:
            raise ValueError("Cannot process order with zero total")
        self.status = "Processed"
        return True


class Inventory:
    def __init__(self):
        # Dictionary mapping product_id to available quantity
        self.stock = {}

    def add_stock(self, product, quantity):
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
        if product.product_id in self.stock:
            self.stock[product.product_id] += quantity
        else:
            self.stock[product.product_id] = quantity

    def remove_stock(self, product, quantity):
        if product.product_id not in self.stock or self.stock[product.product_id] < quantity:
            raise ValueError("Insufficient stock")
        self.stock[product.product_id] -= quantity

    def check_stock(self, product):
        return self.stock.get(product.product_id, 0)


class Coupon:
    def __init__(self, code, discount_rate):
        if discount_rate < 0 or discount_rate > 100:
            raise ValueError("Invalid discount rate")
        self.code = code
        self.discount_rate = discount_rate

    def apply_coupon(self, cart):
        return cart.apply_discount(self.discount_rate)
