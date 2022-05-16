function angle = calcAngle(vec1,vec2)
        
    % ОПИСАНИЕ:
    % Возвращает угол между векторами в прямоугольной декартовой системе
    % координат.
    %
    % ВХОДНЫЕ ДАННЫЕ:
    % vec1 - векторе номер 1  
    % vec2 - векторе номер 2  .
    %
    % ВЫХОДНЫЕ ЗНАЧЕНИЯ:
    % angle - угол в радинах в пределах от -pi до pi.

    temp = dot(vec1,vec2) / (norm(vec1)*norm(vec2));
    if abs( temp ) > 1.0
        temp= sign(temp) * 1.0;
    end 
    angle = acos( temp );
end
        
