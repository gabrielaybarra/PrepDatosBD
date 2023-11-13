from distutils.core import setup
setup(
  name = 'PrepDatosBD',         # How you named your package folder (MyLib)
  packages = ['PrepDatosBD'],   # Chose the same as "name"
  version = '0.1.1',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'This library aims to simplify and agilize the process of data preprocessing and cleaning, which is critical in any data analysis or machine learning project. By providing a variety of tools and functions, users can work more efficiently and ensure the quality of the data they are working with.',   # Give a short description about your library
  author = 'Nerea Zuaznabar & Gabriela Ybarra',                   # Type in your name
  author_email = 'nerea.zuaznabar@alumni.mondragon.edu, gabriela.ybarra@alumni.mondragon.edu',      # Type in your E-Mail
  long_description=open("../README.md", encoding='utf-8').read(),
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/gabrielaybarra/PrepDatosBD.git',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/gabrielaybarra/PrepDatosBD/archive/refs/tags/0.1.1.tar.gz',    # I explain this later on
  keywords = ['Visualize', 'Preprocess', 'Read'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'pandas',
        'matplotlib',
        'scikit-learn',
        'seaborn',
        'openpyxl',
        'xlrd',
        'numpy'
      ],
  classifiers=[
    'Development Status :: 5 - Production/Stable',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)