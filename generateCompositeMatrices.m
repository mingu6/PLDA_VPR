function [mup, A, Sp] = generateCompositeMatrices(x, mu, F, G, Sigma)
[N, d] = size(x);
mup = repmat(mu, N, 1);
A = [repmat(F, N, 1), kron(eye(N), G)];
Sp = repmat(Sigma, N, 1);
end