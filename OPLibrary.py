import math
from scipy.stats import norm
from scipy.optimize import brentq

class BlackScholes: 
    @staticmethod
    def d1(S, K, T, r, sigma):
        return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))

    @staticmethod
    def d2(S, K, T, r, sigma):
        return BlackScholes.d1(S, K, T, r, sigma) - sigma * math.sqrt(T)

    @staticmethod
    def call_price(S, K, T, r, sigma):
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)

    @staticmethod
    def put_price(S, K, T, r, sigma):
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    @staticmethod
    def implied_volatility(option_price, S, K, T, r, option_type='call'):
        def f(sigma):
            price = BlackScholes.call_price(S, K, T, r, sigma) if option_type == 'call' else BlackScholes.put_price(S, K, T, r, sigma)
            return price - option_price

        return brentq(f, 1e-6, 5.0)


class BinomialTree:
    @staticmethod
    def price(S, K, T, r, sigma, steps, option_type='call', american=False):
        dt = T / steps
        u = math.exp(sigma * math.sqrt(dt))
        d = 1 / u
        p = (math.exp(r * dt) - d) / (u - d)
        discount = math.exp(-r * dt)

        prices = [S * (u ** j) * (d ** (steps - j)) for j in range(steps + 1)]
        if option_type == 'call':
            values = [max(price - K, 0) for price in prices]
        else:
            values = [max(K - price, 0) for price in prices]

        for i in range(steps - 1, -1, -1):
            for j in range(i + 1):
                prices[j] = prices[j] / u
                values[j] = discount * (p * values[j + 1] + (1 - p) * values[j])
                if american:
                    exercise = max(0, prices[j] - K if option_type == 'call' else K - prices[j])
                    values[j] = max(values[j], exercise)
        return values[0]
