
%--------------------------------------------------------------------------------
% This demo is included in
% Canonical neural networks perform active inference
% Takuya Isomura et al
%
% Copyright (C) 2020 Takuya Isomura
% (RIKEN Center for Brain Science)
%
% 2020-09-02
%--------------------------------------------------------------------------------

function qdmat = qd_matrix(qd,nv)

qdmat = zeros(nv,nv);
if (length(qd) == 4)
for k1 = 1:4
  k = k1;
  i = (nv+1)/2 - (k1 == 1) + (k1 == 2);
  j = (nv+1)/2 - (k1 == 3) + (k1 == 4);
  qdmat(i,j) = max(qdmat(i,j), qd(k));
end
end

if (length(qd) == 4^2)
for k1 = 1:4
  for k2 = 1:4
    k = (k1-1)*4 + k2;
    i = (nv+1)/2 - (k1 == 1) - (k2 == 1) + (k1 == 2) + (k2 == 2);
    j = (nv+1)/2 - (k1 == 3) - (k2 == 3) + (k1 == 4) + (k2 == 4);
    qdmat(i,j) = max(qdmat(i,j), qd(k));
  end
end
end

if (length(qd) == 4^3)
for k1 = 1:4
  for k2 = 1:4
    for k3 = 1:4
      k = (k1-1)*4^2 + (k2-1)*4 + k3;
      i = (nv+1)/2 - (k1 == 1) - (k2 == 1) - (k3 == 1) + (k1 == 2) + (k2 == 2) + (k3 == 2);
      j = (nv+1)/2 - (k1 == 3) - (k2 == 3) - (k3 == 3) + (k1 == 4) + (k2 == 4) + (k3 == 4);
      qdmat(i,j) = max(qdmat(i,j), qd(k));
    end
  end
end
end

if (length(qd) == 4^4)
for k1 = 1:4
  for k2 = 1:4
    for k3 = 1:4
      for k4 = 1:4
        k = (k1-1)*4^3 + (k2-1)*4^2 + (k3-1)*4 + k4;
        i = (nv+1)/2 - (k1 == 1) - (k2 == 1) - (k3 == 1) - (k4 == 1) + (k1 == 2) + (k2 == 2) + (k3 == 2) + (k4 == 2);
        j = (nv+1)/2 - (k1 == 3) - (k2 == 3) - (k3 == 3) - (k4 == 3) + (k1 == 4) + (k2 == 4) + (k3 == 4) + (k4 == 4);
        qdmat(i,j) = max(qdmat(i,j), qd(k));
      end
    end
  end
end
end

end

