function d = logmvnpdf(x, mu, ld, R)
k = length(mu);
opts.LT = true;
opts1.UT = true;
d = -k / 2 * log(2 * pi) - 1 / 2 * ld - 1 / 2 * (x - mu)' * (linsolve(R, linsolve(R', x - mu, opts), opts1));
end