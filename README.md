EN - this repository contains the code used by group MekaMind during the Robotekin Hackathon 2026. It features 2 main folders:
 /files ------- contains the main code (main.py) which when executed allows you to choose bbetween a variety of functions, them being stacking up the cubes and returning them to their initial position, 
                or building a tower with the cubes (also being able to return them to the initial position). Said cubes are diferentiated by color, recieving the position of each cube. 
                ----- colors.csv : contains the HEX codes for the chromatic gamma used in this project.
                ----- posiciones.config.json : contains the coords (xmin, ymin, xmax, ymax) for each possible position of the cubes.

/lerobot ------ you will need to set the env of lerobot to use the main command we've used in the main script(lerobot-replay), the folder is empty since it will be automatically created when you 
                install the dependencies.

================================================================================================================================================================================================================

ESP- este repositorio contiene el código utilizado por el grupo MekaMind durante la Robotekin Hackathon 2026. Cuenta con 2 carpetas principales:
 /files ------- contiene el código principal (main.py) que, al ejecutarlo, te permite elegir entre una variedad de funciones, las cuales son apilar los cubos y devolverlos a su posición inicial,
                o construir una torre con los cubos (pudiendo también devolverlos a la posición inicial). Dichos cubos se diferencian por color, recibiendo la posición de cada cubo.
                ----- colors.csv : contiene los códigos HEX de la gama cromática utilizada en este proyecto.
                ----- posiciones.config.json : contiene las coordenadas (xmin, ymin, xmax, ymax) para cada posición posible de los cubos.

/lerobot ------ necesitarás crear el entorno de lerobot para utilizar el comando principal que hemos empleado en el script principal (lerobot-replay); la carpeta está vacía ya que se creará automáticamente cuando 
                instales las dependencias.
