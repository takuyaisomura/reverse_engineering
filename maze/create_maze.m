
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

function Map = create_maze(nx, ny)

Map = zeros(ny,nx,'single');
Map([1 ny],:) = 1;
Map(:,[1 nx]) = 1;
Map((ny+1)/2,nx) = 0;
for i = 3:2:ny-2
  for j = 3:2:nx-2
    Map(i,j) = 1;
  end
end
for i = 3:2:ny-2
  for j = 3:2:nx-2
    while 1
      rnd = randi([1 4]);
      if (rnd == 1 && Map(i-1,j) == 0), Map(i-1,j) = 1; break, end
      if (rnd == 2 && Map(i+1,j) == 0), Map(i+1,j) = 1; break, end
      if (rnd == 3 && Map(i,j-1) == 0), Map(i,j-1) = 1; break, end
      if (rnd == 4 && Map(i,j+1) == 0), Map(i,j+1) = 1; break, end
    end
  end
end

