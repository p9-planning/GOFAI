
option(WITH_JAVA "Try to use Java (currently Java 6,7,8)" ON)

if (WITH_JAVA)
#detect java setup, as it is shared between different installations.

find_package(Java 1.8 COMPONENTS Runtime Development)
# find_package(Java COMPONENTS Development)
# find_package(Java COMPONENTS Runtime)
#find_package(JavaLibs)


if (Java_Development_FOUND)

    find_package(JNI)

  if (NOT JNI_FOUND)

set (JAVA_HOME ${JAVA_INCLUDE_PATH}/..)

endif()

  find_package(JNI)

  if (JNI_FOUND)

  include(UseJava)

  #
  #   Java_JAVA_EXECUTABLE      = the full path to the Java runtime
  #   Java_JAVAC_EXECUTABLE     = the full path to the Java compiler
  #   Java_JAVAH_EXECUTABLE     = the full path to the Java header generator
  #   Java_JAVADOC_EXECUTABLE   = the full path to the Java documention generator
  #   Java_IDLJ_EXECUTABLE      = the full path to the Java idl compiler
  #   Java_JAR_EXECUTABLE       = the full path to the Java archiver
  #   Java_JARSIGNER_EXECUTABLE = the full path to the Java jar signer
  #   Java_VERSION_STRING       = Version of java found, eg. 1.6.0_12
  #   Java_VERSION_MAJOR        = The major version of the package found.
  #   Java_VERSION_MINOR        = The minor version of the package found.
  #   Java_VERSION_PATCH        = The patch version of the package found.
  #   Java_VERSION_TWEAK        = The tweak version of the package found (after '_')
  #   Java_VERSION              = This is set to: $major.$minor.$patch(.$tweak)
  #
  # The Java_ADDITIONAL_VERSIONS variable can be used to specify a list
  # of version numbers that should be taken into account when searching
  # for Java.  You need to set this variable before calling
          # find_package(JavaLibs).
  #
  #macro_optional_find_package(JNI ON)
  #   JNI_INCLUDE_DIRS      = the include dirs to use
  #   JNI_LIBRARIES         = the libraries to use
  #   JNI_FOUND             = TRUE if JNI headers and libraries were found.
  #   JAVA_AWT_LIBRARY      = the path to the jawt library
  #   JAVA_JVM_LIBRARY      = the path to the jvm library
  #   JAVA_INCLUDE_PATH     = the include path to jni.h
  #   JAVA_INCLUDE_PATH2    = the include path to jni_md.h
  #   JAVA_AWT_INCLUDE_PATH = the include path to jawt.h
endif (JNI_FOUND)


endif (Java_Development_FOUND)
endif(WITH_JAVA)
