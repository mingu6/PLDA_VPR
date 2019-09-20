function [ey, eyyp] = Estep(x, mu, F, G, Sigma)
[mup, A, Sp] = generateCompositeMatrices(x, mu, F, G, Sigma);
[N, d] = size(A);
X = inv(A' .* repmat(1 ./ Sp', d, 1) * A + eye(d));
ey = X * (A' .* repmat(1 ./ Sp', d, 1)) * (reshape(x', [], 1) - mup);
eyyp = X + ey * ey';
end