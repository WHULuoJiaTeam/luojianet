/*
* StringReplacer.h
*
*  Created on: Sep 24, 2017
*      Author: HeMe2
*
* @brief
* replace sub strings in an c++ std::string.
*/

#ifndef STRINGREPLACER_H_
#define STRINGREPLACER_H_

#include <string>

using std::string;

namespace luojianet_ms {

  class StringReplacer {
  public:
    /**
    * Replaces the provided substring with the replacement string in the origin string.
    * The replacement may be an empty string.
    *
    * @param origin the string to manipulate
    * @param toReplace string that will be replaced
    * @param replacement string that will be inserted instead of toReplace
    * @return The origin string with the defined replacements as a new string.
    */
    string replaceSubstrings(const string &origin, const string & toReplace, const string & replacement);

    /**
    * Searches for a substring byte by byte and returns its start position byte number in the original string.
    * If the toSearch string is not contained in the origin string origin.size() is returned. Only the first
    * substring that is contained, will be considered.
    * @param origin The string to search in
    * @param toSearcfh the string to search in the origin.
    * @return byte position of the start of the substring in the origin string or origin.size() if not contained
    */
    size_t searchForSubstring(const string &origin, const string & toSearch);

    /**
    * Returns if the two strings match in the number of given bytes.
    * Strings do not necessarily need to be null terminated, but in case they are
    * not the result may be inaccurate.
    * @param one string to compare
    * @param two string to compare
    * @return true if each byte matches
    */
    bool matches(const char * one, const char * two, size_t bytes);
  };

}

#endif /* STRINGREPLACER_H_ */